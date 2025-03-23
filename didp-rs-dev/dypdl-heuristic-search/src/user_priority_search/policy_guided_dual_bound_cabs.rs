use super::user_evaluators::PolicyEvaluators;
use crate::f_evaluator_type::FEvaluatorType;
use crate::search_algorithm::data_structure::StateInformation;
use crate::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, FNode, PolicyNode, Search, SearchInput,
    StateInRegistry, SuccessorGenerator, TransitionWithId,
};
use dypdl::{variable_type, ReduceFunction};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a Complete Anytime Beam Search (CABS) solver using the dual bound as a heuristic function.
///
/// It iterates beam search with exponentially increasing beam width.
/// `beam_size` specifies the initial beam width.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `bound_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// It uses the dual bound defined in the DyPDL model as a heuristic function and compute the f-value using a user-provided function
/// that takes the log probability computed by the policy, the g-value, and the h-value.
pub fn create_policy_guided_dual_bound_cabs<T, P, F>(
    model: Rc<dypdl::Model>,
    parameters: CabsParameters<T>,
    bound_evaluator_type: FEvaluatorType,
    policy_evaluators: PolicyEvaluators<P, F>,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
    P: Fn(&StateInRegistry) -> Vec<f64> + 'static,
    F: Fn(f64, T, Option<T>) -> f64 + 'static,
{
    let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| bound_evaluator_type.eval(cost, base_cost);
    let cost = match bound_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    if model.has_dual_bounds() {
        let h_model = model.clone();
        let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| bound_evaluator_type.eval(g, h);
        let node = FNode::generate_root_node(
            model.target.clone(),
            cost,
            &model,
            &h_evaluator,
            &f_evaluator,
            parameters.beam_search_parameters.parameters.primal_bound,
        );
        let node = node.map(|node| {
            let g = node.cost(&model);
            let h = if model.reduce_function == ReduceFunction::Max {
                node.h
            } else {
                -node.h
            };

            PolicyNode::generate_root_node(node, g, Some(h), &policy_evaluators.f_evaluator)
        });
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            move |node: &PolicyNode<_, FNode<_, _>>, transition, primal_bound| {
                let successor = node.node.generate_successor_node(
                    transition,
                    &model,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )?;
                let g = node.cost(&model);
                let h = if model.reduce_function == ReduceFunction::Max {
                    successor.h
                } else {
                    -successor.h
                };

                Some(node.generate_successor_node(
                    successor,
                    g,
                    Some(h),
                    &policy_evaluators.policy,
                    &policy_evaluators.f_evaluator,
                    policy_evaluators.no_accumulation,
                ))
            };
        let beam_search = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };
        Box::new(Cabs::<_, _, _, _>::new(input, beam_search, parameters))
    } else {
        let node = CostNode::generate_root_node(model.target.clone(), cost, &model);
        let g = node.cost(&model);
        let node = PolicyNode::generate_root_node(node, g, None, &policy_evaluators.f_evaluator);
        let input = SearchInput {
            node: Some(node),
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator = move |node: &PolicyNode<_, CostNode<_, _>>, transition, _| {
            let successor = node.node.generate_successor_node(transition, &model)?;
            let g = node.cost(&model);

            Some(node.generate_successor_node(
                successor,
                g,
                None,
                &policy_evaluators.policy,
                &policy_evaluators.f_evaluator,
                policy_evaluators.no_accumulation,
            ))
        };
        let beam_search = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };
        Box::new(Cabs::<_, _, _, _>::new(input, beam_search, parameters))
    }
}
