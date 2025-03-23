use super::user_evaluators::{PolicyEvaluators, UserEvaluators};
use crate::f_evaluator_type::FEvaluatorType;
use crate::search_algorithm::data_structure::{
    CreateTransitionChain, GetTransitions, RcChain, StateInformation,
};
use crate::search_algorithm::{
    Apps, CostNode, FNode, Parameters, PolicyNode, ProgressiveSearchParameters, Search,
    SearchInput, StateInRegistry, SuccessorGenerator, TransitionWithId, UserFNode,
};
use dypdl::variable_type;
use std::fmt;
use std::rc::Rc;
use std::str;

type GEvaluator<U> = dyn Fn(U, &StateInRegistry) -> U;

/// Creates an Anytime Pack Progressive Search (APPS) solver with a user-defined policy and heuristic function.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `bound_evaluator_type` must be specified appropriately according to the cost expressions.
pub fn create_policy_guided_user_priority_apps<T, U, H, P, F>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    bound_evaluator_type: FEvaluatorType,
    user_evaluators: UserEvaluators<Box<GEvaluator<U>>, H>,
    policy_evaluators: PolicyEvaluators<P, F>,
    progressive_parameters: ProgressiveSearchParameters,
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
            parameters.primal_bound,
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
        let transition_evaluator = move |node: &PolicyNode<_, UserFNode<_, _, FNode<_, _>>>,
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
                    other.map(|other: &UserFNode<_, _, FNode<_, _>>| &other.node),
                )?;
                let transition_chain =
                    Rc::from(RcChain::new(node.node.node.transition_chain(), transition));
                let successor = CostNode::new(state, cost, &model, Some(transition_chain));
                let successor = FNode::with_node_and_h_and_f(successor, h, f);
                let g = g_evaluator(successor.last().unwrap(), node.node.g, node.state());
                let (h, f) = UserFNode::evaluate_state(
                    successor.state(),
                    g,
                    &h_evaluator,
                    &f_evaluator,
                    other,
                )?;
                let successor = UserFNode::new(successor, g, h, f);

                Some((successor, g, Some(h)))
            };
            node.insert_successor_node(
                transition,
                registry,
                constructor,
                &policy_evaluators.policy,
                &policy_evaluators.f_evaluator,
                policy_evaluators.no_accumulation,
            )
        };

        Box::new(Apps::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
            progressive_parameters,
        ))
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
        let transition_evaluator = move |node: &PolicyNode<_, UserFNode<_, _, CostNode<_, _>>>,
                                         transition,
                                         registry: &mut _,
                                         _| {
            let constructor = |state, cost, transition, other: Option<&_>| {
                let transition_chain =
                    Rc::from(RcChain::new(node.node.node.transition_chain(), transition));
                let successor = CostNode::new(state, cost, &model, Some(transition_chain));
                let g = g_evaluator(successor.last().unwrap(), node.node.g, node.state());
                let (h, f) = UserFNode::evaluate_state(
                    successor.state(),
                    g,
                    &h_evaluator,
                    &f_evaluator,
                    other,
                )?;
                let successor = UserFNode::new(successor, g, h, f);

                Some((successor, g, Some(h)))
            };
            node.insert_successor_node(
                transition,
                registry,
                constructor,
                &policy_evaluators.policy,
                &policy_evaluators.f_evaluator,
                policy_evaluators.no_accumulation,
            )
        };
        Box::new(Apps::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
            progressive_parameters,
        ))
    }
}
