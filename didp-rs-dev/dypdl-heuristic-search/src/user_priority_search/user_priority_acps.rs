use super::user_evaluators::UserEvaluators;
use crate::f_evaluator_type::FEvaluatorType;
use crate::search_algorithm::data_structure::{CreateTransitionChain, RcChain};
use crate::search_algorithm::{
    Acps, CostNode, FNode, Parameters, ProgressiveSearchParameters, Search, SearchInput,
    StateInRegistry, SuccessorGenerator, TransitionWithId, UserFNode,
};
use dypdl::variable_type;
use std::fmt;
use std::rc::Rc;
use std::str;

type GEvaluator<U> = dyn Fn(U, &StateInRegistry) -> U;

/// Creates an Anytime Column Progressive Search (ACPS) solver using a user-provided heuristic function.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `bound_evaluator_type` must be specified appropriately according to the cost expressions.
pub fn create_user_priority_acps<T, U, H>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    bound_evaluator_type: FEvaluatorType,
    user_evaluators: UserEvaluators<Box<GEvaluator<U>>, H>,
    progressive_parameters: ProgressiveSearchParameters,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
    U: variable_type::Numeric + Ord + 'static,
    H: Fn(&StateInRegistry) -> Option<U> + 'static,
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
    let user_g = match user_evaluators.f_evaluator_type {
        FEvaluatorType::Plus => U::zero(),
        FEvaluatorType::Product => U::one(),
        FEvaluatorType::Max => U::min_value(),
        FEvaluatorType::Min => U::max_value(),
        FEvaluatorType::Overwrite => U::zero(),
    };

    let g_evaluator = move |transition: &TransitionWithId, g, state: &_| {
        if transition.forced {
            user_evaluators.forced_g_evaluators[transition.id](g, state)
        } else {
            user_evaluators.g_evaluators[transition.id](g, state)
        }
    };
    let h_evaluator = user_evaluators.h_evaluator;
    let f_evaluator =
        move |g: U, h: U, _: &StateInRegistry| user_evaluators.f_evaluator_type.eval(g, h);

    if model.has_dual_bounds() {
        let eta_model = model.clone();
        let eta_evaluator = move |state: &_| eta_model.eval_dual_bound(state);
        let bound_evaluator = move |g, eta, _: &_| bound_evaluator_type.eval(g, eta);

        let node = FNode::generate_root_node(
            model.target.clone(),
            cost,
            &model,
            &eta_evaluator,
            &bound_evaluator,
            parameters.primal_bound,
        );
        let node = node.and_then(|node| {
            UserFNode::generate_root_node(node, user_g, &h_evaluator, &f_evaluator)
        });
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator = move |node: &UserFNode<_, _, FNode<_, _>>,
                                         transition,
                                         registry: &mut _,
                                         primal_bound| {
            let constructor = |state, cost, transition, other: Option<&_>| {
                let (h, f) = FNode::evaluate_state(
                    &state,
                    cost,
                    &model,
                    &eta_evaluator,
                    &bound_evaluator,
                    primal_bound,
                    other,
                )?;
                let transition_chain =
                    Rc::from(RcChain::new(node.node.transition_chain(), transition));
                let successor = CostNode::new(state, cost, &model, Some(transition_chain));

                Some(FNode::with_node_and_h_and_f(successor, h, f))
            };
            node.insert_successor_node(
                transition,
                registry,
                constructor,
                &g_evaluator,
                &h_evaluator,
                &f_evaluator,
            )
        };

        Box::new(Acps::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
            progressive_parameters,
        ))
    } else {
        let node = CostNode::generate_root_node(model.target.clone(), cost, &model);
        let node = UserFNode::generate_root_node(node, user_g, &h_evaluator, &f_evaluator);
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            move |node: &UserFNode<_, _, CostNode<_, _>>, transition, registry: &mut _, _| {
                let constructor = |state, cost, transition, _: Option<&_>| {
                    let transition_chain =
                        Rc::from(RcChain::new(node.node.transition_chain(), transition));
                    Some(CostNode::new(state, cost, &model, Some(transition_chain)))
                };
                node.insert_successor_node(
                    transition,
                    registry,
                    constructor,
                    &g_evaluator,
                    &h_evaluator,
                    &f_evaluator,
                )
            };
        Box::new(Acps::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
            progressive_parameters,
        ))
    }
}
