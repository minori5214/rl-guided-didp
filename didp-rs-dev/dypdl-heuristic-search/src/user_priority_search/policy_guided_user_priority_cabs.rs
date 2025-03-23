use super::user_evaluators::{PolicyEvaluators, UserEvaluators};
use crate::f_evaluator_type::FEvaluatorType;
use crate::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, FNode, PolicyNode, Search, SearchInput,
    StateInRegistry, SuccessorGenerator, TransitionWithId, UserFNode,
};
use dypdl::variable_type;
use std::fmt;
use std::rc::Rc;
use std::str;

type GEvaluator<U> = dyn Fn(U, &StateInRegistry) -> U;

pub fn create_policy_guided_user_priority_cabs<T, U, H, P, F>(
    model: Rc<dypdl::Model>,
    parameters: CabsParameters<T>,
    bound_evaluator_type: FEvaluatorType,
    user_evaluators: UserEvaluators<Box<GEvaluator<U>>, H>,
    policy_evaluators: PolicyEvaluators<P, F>,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
    U: variable_type::Numeric + Ord + 'static,
    H: Fn(&StateInRegistry) -> Option<U> + 'static,
    P: Fn(&StateInRegistry) -> Vec<f64> + 'static,
    F: Fn(f64, U, Option<U>) -> f64 + 'static,
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
            parameters.beam_search_parameters.parameters.primal_bound,
        );
        let node = node.and_then(|node| {
            UserFNode::generate_root_node(node, user_g, &h_evaluator, &f_evaluator)
        });
        let node = node.map(|node| {
            let g = node.g;
            let h = node.h;
            PolicyNode::generate_root_node(node, g, Some(h), &policy_evaluators.f_evaluator)
        });
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            move |node: &PolicyNode<_, UserFNode<_, _, FNode<_, _>>>, transition, primal_bound| {
                let successor = node.node.node.generate_successor_node(
                    transition,
                    &model,
                    &eta_evaluator,
                    &bound_evaluator,
                    primal_bound,
                )?;
                let successor = node.node.generate_successor_node(
                    successor,
                    &g_evaluator,
                    &h_evaluator,
                    &f_evaluator,
                )?;
                let g = successor.g;
                let h = successor.h;

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
        let node = UserFNode::generate_root_node(node, user_g, &h_evaluator, &f_evaluator);
        let node = node.map(|node| {
            let g = node.g;
            let h = node.h;

            PolicyNode::generate_root_node(node, g, Some(h), &policy_evaluators.f_evaluator)
        });
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            move |node: &PolicyNode<_, UserFNode<_, _, CostNode<_, _>>>, transition, _| {
                let successor = node.node.node.generate_successor_node(transition, &model)?;
                let successor = node.node.generate_successor_node(
                    successor,
                    &g_evaluator,
                    &h_evaluator,
                    &f_evaluator,
                )?;
                let g = successor.g;
                let h = successor.h;

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
    }
}
