use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition::TransitionWithId;
use super::super::transition_chain::GetTransitions;
use super::BfsNode;
use dypdl::variable_type::{Continuous, Numeric, OrderedContinuous};
use dypdl::{Model, Transition, TransitionInterface};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by the f-value computed with a policy.
/// The node having the lowest f-value has the highest priority.
/// Ties are broken by the underlying node.
#[derive(Debug, Clone)]
pub struct PolicyNode<T, N, V = Transition> {
    /// f-value.
    pub f: OrderedContinuous,
    /// Log probability computed by a policy.
    pub log_pi: f64,
    /// Underlying node.
    pub node: N,
    log_probabilities: RefCell<Option<Vec<f64>>>,
    _phantom: PhantomData<(T, V)>,
}

impl<T, N, V> PartialEq for PolicyNode<T, N, V>
where
    N: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.node == other.node
    }
}

impl<T, N, V> Eq for PolicyNode<T, N, V> where N: Eq {}

impl<T, N, V> Ord for PolicyNode<T, N, V>
where
    N: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match other.f.cmp(&self.f) {
            Ordering::Equal => self.node.cmp(&other.node),
            result => result,
        }
    }
}

impl<T, N, V> PartialOrd for PolicyNode<T, N, V>
where
    N: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, N, V> StateInformation<T> for PolicyNode<T, N, V>
where
    T: Numeric,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    N: StateInformation<T> + GetTransitions<TransitionWithId<V>>,
{
    #[inline]
    fn state(&self) -> &StateInRegistry {
        self.node.state()
    }

    #[inline]
    fn state_mut(&mut self) -> &mut StateInRegistry {
        self.node.state_mut()
    }

    #[inline]
    fn cost(&self, model: &Model) -> T {
        self.node.cost(model)
    }

    #[inline]
    fn bound(&self, model: &Model) -> Option<T> {
        self.node.bound(model)
    }

    #[inline]
    fn is_closed(&self) -> bool {
        self.node.is_closed()
    }

    #[inline]
    fn close(&self) {
        self.node.close()
    }
}

impl<T, N, V> GetTransitions<TransitionWithId<V>> for PolicyNode<T, N, V>
where
    T: Numeric,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    N: StateInformation<T> + GetTransitions<TransitionWithId<V>>,
{
    #[inline]
    fn transitions(&self) -> Vec<TransitionWithId<V>> {
        self.node.transitions()
    }

    #[inline]
    fn last(&self) -> Option<&TransitionWithId<V>> {
        self.node.last()
    }
}

impl<T, N, V> BfsNode<T, TransitionWithId<V>> for PolicyNode<T, N, V>
where
    T: Numeric + Display,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    N: BfsNode<T, TransitionWithId<V>>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        false
    }
}

impl<T, N, V> PolicyNode<T, N, V>
where
    T: Numeric + Display,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    N: StateInformation<T> + GetTransitions<TransitionWithId<V>> + Ord,
{
    /// Creates a new node.
    pub fn new(node: N, f: OrderedContinuous, log_pi: Continuous) -> Self {
        Self {
            node,
            f,
            log_pi,
            log_probabilities: RefCell::new(None),
            _phantom: PhantomData,
        }
    }

    /// Creates a root node.
    pub fn generate_root_node<U, F>(node: N, g: U, h: Option<U>, f_evaluator: F) -> Self
    where
        F: FnOnce(f64, U, Option<U>) -> f64,
    {
        let priority = OrderedContinuous::from(f_evaluator(0.0, g, h));
        Self::new(node, priority, 0.0)
    }

    /// Evaluates the policy and computes the f-value.
    /// `f_evaluator` takes the log probability, g-value and h-value as arguments.
    /// if `no_accumulation`, the log probability is not accumulated along the path.
    pub fn evaluate_policy<U, P, F>(
        &self,
        transition: &TransitionWithId<V>,
        g: U,
        h: Option<U>,
        policy: P,
        f_evaluator: F,
        no_accumulation: bool,
    ) -> (OrderedContinuous, Continuous)
    where
        P: FnOnce(&StateInRegistry) -> Vec<f64>,
        F: FnOnce(f64, U, Option<U>) -> f64,
    {
        let log_p = if transition.forced {
            0.0
        } else {
            if self.log_probabilities.borrow().is_none() {
                *self.log_probabilities.borrow_mut() = Some(policy(self.state()));
            }

            self.log_probabilities.borrow().as_ref().unwrap()[transition.id]
        };
        let log_pi = if no_accumulation {
            log_p
        } else {
            self.log_pi + log_p
        };
        let f = OrderedContinuous::from(f_evaluator(log_pi, g, h));

        (f, log_pi)
    }

    /// Generates a successor node.
    /// `f_evaluator` takes the log probability, g-value and h-value as arguments.
    /// if `no_accumulation`, the log probability is not accumulated along the path.
    pub fn generate_successor_node<U, P, F>(
        &self,
        node: N,
        g: U,
        h: Option<U>,
        policy: P,
        f_evaluator: F,
        no_accumulation: bool,
    ) -> Self
    where
        P: FnOnce(&StateInRegistry) -> Vec<f64>,
        F: FnOnce(f64, U, Option<U>) -> f64,
    {
        let transition = node.last().unwrap();
        let (f, log_pi) =
            self.evaluate_policy(transition, g, h, policy, f_evaluator, no_accumulation);

        Self::new(node, f, log_pi)
    }

    /// Generates a successor node and insert it into the state registry.
    /// `f_evaluator` takes the log probability, g-value and h-value as arguments.
    /// if `no_accumulation`, the log probability is not accumulated along the path.
    pub fn insert_successor_node<R, C, U, P, F>(
        &self,
        transition: R,
        registry: &mut StateRegistry<T, Self>,
        constructor: C,
        policy: P,
        f_evaluator: F,
        no_accumulation: bool,
    ) -> Option<(Rc<Self>, bool)>
    where
        R: Deref<Target = TransitionWithId<V>>,
        C: FnOnce(StateInRegistry, T, R, Option<&N>) -> Option<(N, U, Option<U>)>,
        P: FnOnce(&StateInRegistry) -> Vec<f64>,
        F: FnOnce(f64, U, Option<U>) -> f64,
    {
        let (state, cost) = registry.model().generate_successor_state(
            self.state(),
            self.cost(registry.model()),
            transition.deref(),
            None,
        )?;
        let constructor = |state, cost, other: Option<&Self>| {
            let (node, g, h) = constructor(state, cost, transition, other.map(|n| &n.node))?;
            let (f, log_pi) = self.evaluate_policy(
                node.last().unwrap(),
                g,
                h,
                policy,
                f_evaluator,
                no_accumulation,
            );
            let mut node = Self::new(node, f, log_pi);

            if let Some(other) = other {
                if other.log_probabilities.borrow().is_some() {
                    node.log_probabilities = other.log_probabilities.clone();
                }
            }

            Some(node)
        };

        let (successor, dominated) = registry.insert_with(state, cost, constructor)?;

        let mut generated = true;

        if let Some(dominated) = dominated {
            if !dominated.is_closed() {
                dominated.close();
                generated = false;
            }
        }

        Some((successor, generated))
    }
}

#[cfg(test)]
mod tests {
    use dypdl::prelude::*;
    use dypdl::{variable_type::OrderedContinuous, AddEffect, Integer};

    use super::*;
    use crate::search_algorithm::{
        data_structure::{CreateTransitionChain, RcChain},
        CostNode,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let mut node = PolicyNode::new(cost_node.clone(), OrderedContinuous::from(2.0), -0.3);

        assert_eq!(node.f, OrderedContinuous::from(2.0));
        assert_eq!(node.log_pi, -0.3);
        assert_eq!(node.node, cost_node);
        assert_eq!(node.node.transition_chain(), None);
        assert_eq!(node.state(), &state);
        assert_eq!(node.state_mut(), &mut state);
        assert_eq!(node.cost(&model), 2);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
    }

    #[test]
    fn test_close() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let node = PolicyNode::new(cost_node.clone(), OrderedContinuous::from(2.0), -0.3);

        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_get_transition() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let transition1 = TransitionWithId {
            transition: Transition::new("t1"),
            id: 0,
            forced: false,
        };
        let transition2 = TransitionWithId {
            transition: Transition::new("t2"),
            id: 0,
            forced: false,
        };
        let cost = 10;
        let transition_chain = Some(Rc::new(RcChain::new(None, Rc::new(transition1.clone()))));
        let transition_chain = Some(Rc::new(RcChain::new(
            transition_chain,
            Rc::new(transition2.clone()),
        )));
        let node =
            CostNode::<_, TransitionWithId>::new(state, cost, &model, transition_chain.clone());
        let node = PolicyNode::new(node.clone(), OrderedContinuous::from(2.0), -0.3);

        assert_eq!(node.last(), Some(&transition2));
        assert_eq!(node.transitions(), vec![transition1, transition2]);
        assert_eq!(node.node.transition_chain(), transition_chain);
    }

    #[test]
    fn test_ord() {
        let model = Model::default();

        let node1 = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            1,
            &model,
            None,
        );
        let node1 = PolicyNode::new(node1, OrderedContinuous::from(2.0), -0.3);
        let node2 = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            0,
            &model,
            None,
        );
        let node2 = PolicyNode::new(node2, OrderedContinuous::from(2.2), -0.4);
        let node3 = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            2,
            &model,
            None,
        );
        let node3 = PolicyNode::new(node3, OrderedContinuous::from(2.0), -0.2);

        assert!(node1 == node1);
        assert!(node1 <= node1);
        assert!(node1 >= node1);
        assert!(node1 != node2);
        assert!(node1 >= node1);
        assert!(node1 >= node2);
        assert!(node1 > node2);
        assert!(node2 <= node1);
        assert!(node2 < node1);
        assert!(node1 != node3);
        assert!(node1 >= node3);
        assert!(node1 > node3);
        assert!(node3 <= node1);
        assert!(node3 < node1);
    }

    #[test]
    fn test_ordered_by_bound() {
        assert!(!PolicyNode::<Integer, CostNode<_, _>>::ordered_by_bound());
    }

    #[test]
    fn test_generate_root_node() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut cost_node = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            1,
            &model,
            None,
        );
        let mut node = PolicyNode::generate_root_node(cost_node.clone(), 1, Some(3), |_, _, _| 1.5);

        assert_eq!(node.f, OrderedContinuous::from(1.5));
        assert_eq!(node.log_pi, 0.0);
        assert_eq!(node.node, cost_node);
        assert_eq!(node.node.transition_chain(), None);
        assert_eq!(node.state(), cost_node.state());
        assert_eq!(node.state_mut(), cost_node.state_mut());
        assert_eq!(node.cost(&model), 1);
        assert_eq!(node.bound(&model), None);
        assert!(!node.is_closed());
        assert_eq!(node.transitions(), vec![]);
        assert_eq!(node.last(), None);
    }

    #[test]
    fn test_evaluate_policy() {
        let model = Model::default();
        let transition1 = TransitionWithId {
            transition: Transition::new("t1"),
            id: 0,
            forced: false,
        };
        let transition2 = TransitionWithId {
            transition: Transition::new("t2"),
            id: 1,
            forced: false,
        };

        let policy = |_: &_| vec![-0.7, -0.1];
        let f_evaluator = |_, _, _| 1.5;

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);

        let (f, log_pi) = node.evaluate_policy(
            &transition1,
            node.cost(&model),
            None,
            policy,
            f_evaluator,
            false,
        );
        assert_eq!(f, OrderedContinuous::from(1.5));
        assert_relative_eq!(log_pi, -1.0);

        let (f, log_pi) = node.evaluate_policy(
            &transition2,
            node.cost(&model),
            None,
            policy,
            f_evaluator,
            false,
        );
        assert_eq!(f, OrderedContinuous::from(1.5));
        assert_relative_eq!(log_pi, -0.4);
    }

    #[test]
    fn test_evaluate_policy_forced() {
        let model = Model::default();
        let transition = TransitionWithId {
            transition: Transition::new("t1"),
            id: 0,
            forced: true,
        };

        let policy = |_: &_| vec![-0.7, -0.1];
        let f_evaluator = |_, _, _| 1.5;

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);

        let (f, log_pi) = node.evaluate_policy(
            &transition,
            node.cost(&model),
            None,
            policy,
            f_evaluator,
            false,
        );
        assert_eq!(f, OrderedContinuous::from(1.5));
        assert_relative_eq!(log_pi, -0.3);
    }

    #[test]
    fn test_evaluate_policy_no_accumulate() {
        let model = Model::default();
        let transition1 = TransitionWithId {
            transition: Transition::new("t1"),
            id: 0,
            forced: false,
        };
        let transition2 = TransitionWithId {
            transition: Transition::new("t2"),
            id: 1,
            forced: false,
        };

        let policy = |_: &_| vec![-0.7, -0.1];
        let f_evaluator = |_, _, _| 1.5;

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);

        let (f, log_pi) = node.evaluate_policy(
            &transition1,
            node.cost(&model),
            None,
            policy,
            f_evaluator,
            true,
        );
        assert_eq!(f, OrderedContinuous::from(1.5));
        assert_relative_eq!(log_pi, -0.7);

        let (f, log_pi) = node.evaluate_policy(
            &transition2,
            node.cost(&model),
            None,
            policy,
            f_evaluator,
            true,
        );
        assert_eq!(f, OrderedContinuous::from(1.5));
        assert_relative_eq!(log_pi, -0.1);
    }

    #[test]
    fn test_generate_successor_node() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let mut transition = Transition::new("t1");
        let result = transition.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let transition = TransitionWithId {
            transition,
            id: 0,
            forced: false,
        };

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 1, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);

        let transition_chain = Some(Rc::new(RcChain::new(None, Rc::new(transition.clone()))));
        let mut successor_cost_node =
            CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, transition_chain);

        let policy = |_: &_| vec![0.0];
        let f_evaluator = |_, _, _| 1.5;

        let mut successor = node.generate_successor_node(
            successor_cost_node.clone(),
            node.cost(&model),
            None,
            policy,
            f_evaluator,
            false,
        );

        assert_eq!(successor.f, OrderedContinuous::from(1.5));
        assert_relative_eq!(successor.log_pi, -0.3);
        assert_eq!(successor.node, successor_cost_node);
        assert_eq!(
            successor.node.transition_chain(),
            successor_cost_node.transition_chain()
        );
        assert_eq!(successor.state(), successor_cost_node.state());
        assert_eq!(successor.state_mut(), successor_cost_node.state_mut());
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), successor_cost_node.transitions());
        assert_eq!(successor.last(), successor_cost_node.last());
    }

    #[test]
    fn test_insert_successor_node_non_dominance() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let mut transition = Transition::new("t1");
        let result = transition.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let transition = Rc::new(TransitionWithId {
            transition,
            id: 0,
            forced: false,
        });

        let mut registry = StateRegistry::new(model.clone());

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 1, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let policy = |_: &_| vec![0.0];
        let f_evaluator = |_, _, _| 1.5;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some((
                CostNode::new(state, cost, &model, Some(transition_chain)),
                cost,
                None,
            ))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            policy,
            f_evaluator,
            false,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();

        let expected_state = transition.apply(&state, &model.table_registry);
        let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
        let successor_cost_node =
            CostNode::<_, TransitionWithId>::new(expected_state, 2, &model, Some(transition_chain));

        assert_eq!(successor.f, OrderedContinuous::from(1.5));
        assert_relative_eq!(successor.log_pi, -0.3);
        assert_eq!(successor.node, successor_cost_node);
        assert_eq!(
            successor.node.transition_chain(),
            successor_cost_node.transition_chain()
        );
        assert_eq!(successor.state(), successor_cost_node.state());
        assert_eq!(successor.cost(&model), 2);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), successor_cost_node.transitions());
        assert_eq!(successor.last(), successor_cost_node.last());

        assert!(generated);
        assert!(!node.is_closed())
    }

    #[test]
    fn test_insert_successor_node_dominating() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", false, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let mut transition = Transition::new("t1");
        let result = transition.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost);

        let transition = Rc::new(TransitionWithId {
            transition,
            id: 0,
            forced: false,
        });

        let mut registry = StateRegistry::new(model.clone());

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 1, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let policy = |_: &_| vec![0.0];
        let f_evaluator = |_, _, _| 1.5;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some((
                CostNode::new(state, cost, &model, Some(transition_chain)),
                cost,
                None,
            ))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            policy,
            f_evaluator,
            false,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();

        let expected_state = transition.apply(&state, &model.table_registry);
        let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
        let successor_cost_node =
            CostNode::<_, TransitionWithId>::new(expected_state, 1, &model, Some(transition_chain));

        assert_eq!(successor.f, OrderedContinuous::from(1.5));
        assert_relative_eq!(successor.log_pi, -0.3);
        assert_eq!(successor.node, successor_cost_node);
        assert_eq!(
            successor.node.transition_chain(),
            successor_cost_node.transition_chain()
        );
        assert_eq!(successor.state(), successor_cost_node.state());
        assert_eq!(successor.cost(&model), 1);
        assert_eq!(successor.bound(&model), None);
        assert!(!successor.is_closed());
        assert_eq!(successor.transitions(), successor_cost_node.transitions());
        assert_eq!(successor.last(), successor_cost_node.last());

        assert!(!generated);
        assert!(node.is_closed())
    }

    #[test]
    fn test_insert_successor_node_dominated() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", true, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let model = Rc::new(model);

        let mut transition = Transition::new("t1");
        let result = transition.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let transition = Rc::new(TransitionWithId {
            transition,
            id: 0,
            forced: false,
        });

        let mut registry = StateRegistry::new(model.clone());

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 1, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let policy = |_: &_| vec![0.0];
        let f_evaluator = |_, _, _| 1.5;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some((
                CostNode::new(state, cost, &model, Some(transition_chain)),
                cost,
                None,
            ))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            policy,
            f_evaluator,
            false,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_insert_successor_node_pruned_by_constraints() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();
        let result =
            model.add_state_constraint(Condition::comparison_i(ComparisonOperator::Le, v1 + v2, 3));
        assert!(result.is_ok());
        let model = Rc::new(model);

        let mut transition = Transition::new("t1");
        let result = transition.add_effect(v1, v1 + 2);
        assert!(result.is_ok());
        let result = transition.add_effect(v2, v2 + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);

        let transition = Rc::new(TransitionWithId {
            transition,
            id: 0,
            forced: false,
        });

        let mut registry = StateRegistry::new(model.clone());

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 1, &model, None);
        let node = PolicyNode::new(cost_node, OrderedContinuous::from(2.0), -0.3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let policy = |_: &_| vec![0.0];
        let f_evaluator = |_, _, _| 1.5;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some((
                CostNode::new(state, cost, &model, Some(transition_chain)),
                cost,
                None,
            ))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            policy,
            f_evaluator,
            false,
        );
        assert_eq!(result, None);
    }
}
