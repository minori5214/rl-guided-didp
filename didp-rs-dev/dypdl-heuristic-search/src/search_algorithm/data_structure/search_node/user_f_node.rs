use super::super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::super::transition::TransitionWithId;
use super::super::transition_chain::GetTransitions;
use super::BfsNode;
use dypdl::variable_type::Numeric;
use dypdl::{Model, StateInterface, Transition, TransitionInterface};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

/// Node ordered by the f-value computed by user-provided functions.
/// The node having the lowest f-value has the highest priority.
/// Ties are broken by the h-value, and then the underlying node.
#[derive(Debug, Clone)]
pub struct UserFNode<T, U, N, V = Transition> {
    /// g-value
    pub g: U,
    /// h-value
    pub h: U,
    /// f-value
    pub f: U,
    /// Underlying node
    pub node: N,
    _phantom: PhantomData<(T, V)>,
}

impl<T, U, N, V> PartialEq for UserFNode<T, U, N, V>
where
    U: PartialEq,
    N: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h && self.node == other.node
    }
}

impl<T, U, N, V> Eq for UserFNode<T, U, N, V>
where
    U: Eq,
    N: Eq,
{
}

impl<T, U, N, V> Ord for UserFNode<T, U, N, V>
where
    U: Ord,
    N: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match other.f.cmp(&self.f) {
            Ordering::Equal => match other.h.cmp(&self.h) {
                Ordering::Equal => self.node.cmp(&other.node),
                result => result,
            },
            result => result,
        }
    }
}

impl<T, U, N, V> PartialOrd for UserFNode<T, U, N, V>
where
    U: Ord,
    N: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, U, N, V> StateInformation<T> for UserFNode<T, U, N, V>
where
    T: Numeric,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    U: Ord,
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

impl<T, U, N, V> GetTransitions<TransitionWithId<V>> for UserFNode<T, U, N, V>
where
    T: Numeric,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    U: Ord,
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

impl<T, U, N, V> BfsNode<T, TransitionWithId<V>> for UserFNode<T, U, N, V>
where
    T: Numeric + Display,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    U: Ord,
    N: BfsNode<T, TransitionWithId<V>>,
{
    #[inline]
    fn ordered_by_bound() -> bool {
        false
    }
}

impl<T, U, N, V> UserFNode<T, U, N, V>
where
    T: Numeric + Display,
    V: TransitionInterface + Clone + Debug,
    Transition: From<TransitionWithId<V>>,
    U: Ord + Copy,
    N: StateInformation<T> + GetTransitions<TransitionWithId<V>>,
{
    /// Create a new node.
    pub fn new(node: N, g: U, h: U, f: U) -> Self {
        Self {
            g,
            h,
            f,
            node,
            _phantom: PhantomData,
        }
    }

    /// Evaluate a state given its g-value, h- and f-evaluators, and another node sharing the same state.
    pub fn evaluate_state<S, H, F>(
        state: &S,
        g: U,
        h_evaluator: H,
        f_evaluator: F,
        other: Option<&Self>,
    ) -> Option<(U, U)>
    where
        S: StateInterface,
        H: FnOnce(&S) -> Option<U>,
        F: FnOnce(U, U, &S) -> U,
    {
        let h = if let Some(other) = other {
            other.h
        } else {
            h_evaluator(state)?
        };
        let f = f_evaluator(g, h, state);

        Some((h, f))
    }

    /// Generate a root node.
    pub fn generate_root_node<H, F>(node: N, g: U, h_evaluator: H, f_evaluator: F) -> Option<Self>
    where
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
    {
        let (h, f) = Self::evaluate_state(node.state(), g, h_evaluator, f_evaluator, None)?;

        Some(Self::new(node, g, h, f))
    }

    /// Generate a successor node.
    pub fn generate_successor_node<G, H, F>(
        &self,
        node: N,
        g_evaluator: G,
        h_evaluator: H,
        f_evaluator: F,
    ) -> Option<Self>
    where
        G: FnOnce(&TransitionWithId<V>, U, &StateInRegistry) -> U,
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
    {
        let g = g_evaluator(node.last().unwrap(), self.g, self.state());
        let (h, f) = Self::evaluate_state(node.state(), g, h_evaluator, f_evaluator, None)?;

        Some(Self::new(node, g, h, f))
    }

    /// Generate and insert a successor node into the state registry.
    pub fn insert_successor_node<R, C, G, H, F>(
        &self,
        transition: R,
        registry: &mut StateRegistry<T, Self>,
        constructor: C,
        g_evaluator: G,
        h_evaluator: H,
        f_evaluator: F,
    ) -> Option<(Rc<Self>, bool)>
    where
        R: Deref<Target = TransitionWithId<V>>,
        C: FnOnce(StateInRegistry, T, R, Option<&N>) -> Option<N>,
        G: FnOnce(&TransitionWithId<V>, U, &StateInRegistry) -> U,
        H: FnOnce(&StateInRegistry) -> Option<U>,
        F: FnOnce(U, U, &StateInRegistry) -> U,
    {
        let (state, cost) = registry.model().generate_successor_state(
            self.state(),
            self.cost(registry.model()),
            transition.deref(),
            None,
        )?;
        let constructor = |state, cost, other: Option<&Self>| {
            let node = constructor(state, cost, transition, other.map(|node| &node.node))?;
            let g = g_evaluator(node.last().unwrap(), self.g, self.state());
            let (h, f) = Self::evaluate_state(node.state(), g, h_evaluator, f_evaluator, other)?;

            Some(Self::new(node, g, h, f))
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

    #[test]
    fn test_new() {
        let mut model = Model::default();
        let v1 = model.add_integer_resource_variable("v1", true, 0);
        assert!(v1.is_ok());
        let v2 = model.add_integer_resource_variable("v2", false, 0);
        assert!(v2.is_ok());

        let mut state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let mut node = UserFNode::new(cost_node.clone(), 1, 2, 3);

        assert_eq!(node.g, 1);
        assert_eq!(node.h, 2);
        assert_eq!(node.f, 3);
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
        let node = UserFNode::new(cost_node.clone(), 1, 2, 3);

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
        let node = UserFNode::new(node.clone(), 1, 2, 3);

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
        let node1 = UserFNode::new(node1, 1, 2, 3);
        let node2 = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            0,
            &model,
            None,
        );
        let node2 = UserFNode::new(node2, 3, 1, 4);
        let node3 = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            1,
            &model,
            None,
        );
        let node3 = UserFNode::new(node3, 0, 3, 3);
        let node4 = CostNode::<_, TransitionWithId>::new(
            StateInRegistry::from(model.target.clone()),
            2,
            &model,
            None,
        );
        let node4 = UserFNode::new(node4, 1, 2, 3);

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
        assert!(node1 != node4);
        assert!(node1 >= node4);
        assert!(node1 > node4);
        assert!(node4 <= node1);
        assert!(node4 < node1);
    }

    #[test]
    fn test_ordered_by_bound() {
        assert!(!UserFNode::<Integer, OrderedContinuous, CostNode<_, _>>::ordered_by_bound());
    }

    #[test]
    fn test_evaluate_state_some() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let g = 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;
        let other = None;

        let result = UserFNode::<Integer, _, CostNode<_, TransitionWithId>>::evaluate_state(
            &state,
            g,
            h_evaluator,
            f_evaluator,
            other,
        );
        assert_eq!(result, Some((2, 3)));
    }

    #[test]
    fn test_evaluate_state_some_with_other() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let g = 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;
        let other = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let other = UserFNode::new(other, 2, 2, 4);

        let result = UserFNode::<Integer, _, CostNode<_, TransitionWithId>>::evaluate_state(
            &state,
            g,
            h_evaluator,
            f_evaluator,
            Some(&other),
        );
        assert_eq!(result, Some((2, 3)));
    }

    #[test]
    fn test_evaluate_state_none() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let g = 1;
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g, h, _: &_| g + h;
        let other = None;

        let result = UserFNode::<Integer, _, CostNode<_, TransitionWithId>>::evaluate_state(
            &state,
            g,
            h_evaluator,
            f_evaluator,
            other,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_root_node_some() {
        let model = Model::default();
        let mut state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let g = 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;

        let result = UserFNode::generate_root_node(cost_node.clone(), g, h_evaluator, f_evaluator);
        assert!(result.is_some());
        let mut node = result.unwrap();

        assert_eq!(node.g, 1);
        assert_eq!(node.h, 2);
        assert_eq!(node.f, 3);
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
    fn test_generate_root_node_none() {
        let model = Model::default();
        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, None);
        let g = 1;
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g, h, _: &_| g + h;

        let result = UserFNode::generate_root_node(cost_node.clone(), g, h_evaluator, f_evaluator);
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_successor_node_some() {
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
        let node = UserFNode::new(cost_node, 1, 2, 3);

        let transition_chain = Some(Rc::new(RcChain::new(None, Rc::new(transition.clone()))));
        let mut successor_cost_node =
            CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, transition_chain);

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;

        let result = node.generate_successor_node(
            successor_cost_node.clone(),
            g_evaluator,
            h_evaluator,
            f_evaluator,
        );
        assert!(result.is_some());
        let mut successor = result.unwrap();

        assert_eq!(successor.g, 2);
        assert_eq!(successor.h, 2);
        assert_eq!(successor.f, 4);
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
    fn test_generate_successor_node_none() {
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
        let node = UserFNode::new(cost_node, 1, 2, 3);

        let transition_chain = Some(Rc::new(RcChain::new(None, Rc::new(transition.clone()))));
        let successor_cost_node =
            CostNode::<_, TransitionWithId>::new(state.clone(), 2, &model, transition_chain);

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g, h, _: &_| g + h;

        let result = node.generate_successor_node(
            successor_cost_node,
            g_evaluator,
            h_evaluator,
            f_evaluator,
        );
        assert_eq!(result, None);
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
        let node = UserFNode::new(cost_node, 1, 2, 3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some(CostNode::new(state, cost, &model, Some(transition_chain)))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            g_evaluator,
            h_evaluator,
            f_evaluator,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();

        let expected_state = transition.apply(&state, &model.table_registry);
        let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
        let successor_cost_node =
            CostNode::<_, TransitionWithId>::new(expected_state, 2, &model, Some(transition_chain));

        assert_eq!(successor.g, 2);
        assert_eq!(successor.h, 2);
        assert_eq!(successor.f, 4);
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
        let node = UserFNode::new(cost_node, 1, 2, 3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;
        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some(CostNode::new(state, cost, &model, Some(transition_chain)))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            g_evaluator,
            h_evaluator,
            f_evaluator,
        );
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();

        let expected_state = transition.apply(&state, &model.table_registry);
        let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
        let successor_cost_node =
            CostNode::<_, TransitionWithId>::new(expected_state, 1, &model, Some(transition_chain));

        assert_eq!(successor.g, 2);
        assert_eq!(successor.h, 2);
        assert_eq!(successor.f, 4);
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
        transition.set_cost(IntegerExpression::Cost);

        let transition = Rc::new(TransitionWithId {
            transition,
            id: 0,
            forced: false,
        });

        let mut registry = StateRegistry::new(model.clone());

        let state = StateInRegistry::from(model.target.clone());
        let cost_node = CostNode::<_, TransitionWithId>::new(state.clone(), 1, &model, None);
        let node = UserFNode::new(cost_node, 1, 2, 3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;
        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some(CostNode::new(state, cost, &model, Some(transition_chain)))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            g_evaluator,
            h_evaluator,
            f_evaluator,
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
        let node = UserFNode::new(cost_node, 1, 2, 3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| Some(2);
        let f_evaluator = |g, h, _: &_| g + h;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some(CostNode::new(state, cost, &model, Some(transition_chain)))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            g_evaluator,
            h_evaluator,
            f_evaluator,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_insert_successor_node_pruned_by_h() {
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
        let node = UserFNode::new(cost_node, 1, 2, 3);
        let result = registry.insert(node);
        assert!(result.is_some());
        let (node, _) = result.unwrap();

        let g_evaluator = |_: &_, g, _: &_| g + 1;
        let h_evaluator = |_: &StateInRegistry| None;
        let f_evaluator = |g, h, _: &_| g + h;

        let constructor = |state, cost, transition, _: Option<&_>| {
            let transition_chain = Rc::from(RcChain::new(node.node.transition_chain(), transition));
            Some(CostNode::new(state, cost, &model, Some(transition_chain)))
        };

        let result = node.insert_successor_node(
            transition.clone(),
            &mut registry,
            constructor,
            g_evaluator,
            h_evaluator,
            f_evaluator,
        );
        assert_eq!(result, None);
    }
}
