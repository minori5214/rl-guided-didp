use crate::model::StatePy;
use dypdl::prelude::*;
use dypdl::variable_type::{Numeric, OrderedContinuous};
use dypdl_heuristic_search::search_algorithm::StateInRegistry;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::rc::Rc;

type GEvaluator = Box<dyn Fn(OrderedContinuous, &StateInRegistry) -> OrderedContinuous>;

pub fn create_g_evaluator(g_evaluator: Py<PyAny>) -> GEvaluator {
    Box::new(move |g, state: &_| {
        Python::with_gil(|py| {
            let state = State::from(state.clone());
            let args = PyTuple::new(
                py,
                &[f64::from(g).into_py(py), StatePy::from(state).into_py(py)],
            );
            OrderedContinuous::from(
                g_evaluator
                    .call1(py, args)
                    .unwrap()
                    .extract::<Continuous>(py)
                    .unwrap(),
            )
        })
    })
}

pub fn create_default_g_evaluator_vectors(model: &Rc<Model>) -> (Vec<GEvaluator>, Vec<GEvaluator>) {
    let mut forced_g_evaluator_vec = Vec::with_capacity(model.forward_forced_transitions.len());

    for t in &model.forward_forced_transitions {
        let t = t.clone();
        let model = model.clone();

        forced_g_evaluator_vec.push(Box::new(move |g, state: &_| {
            t.eval_cost(g, state, &model.table_registry)
        }) as GEvaluator);
    }

    let mut g_evaluator_vec = Vec::with_capacity(model.forward_transitions.len());

    for t in &model.forward_transitions {
        let t = t.clone();
        let model = model.clone();

        g_evaluator_vec.push(Box::new(move |g, state: &_| {
            t.eval_cost(g, state, &model.table_registry)
        }) as GEvaluator);
    }

    (forced_g_evaluator_vec, g_evaluator_vec)
}

pub fn create_g_evaluator_vectors(
    model: &Model,
    g_evaluators: &mut HashMap<String, Py<PyAny>>,
) -> PyResult<(Vec<GEvaluator>, Vec<GEvaluator>)> {
    let mut forced_g_evaluator_vec = Vec::with_capacity(model.forward_forced_transitions.len());

    for t in &model.forward_forced_transitions {
        if let Some(g) = g_evaluators.remove(&t.get_full_name()) {
            forced_g_evaluator_vec.push(create_g_evaluator(g));
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "No g evaluator found for forced transition {}",
                t.get_full_name()
            )));
        }
    }

    let mut g_evaluator_vec = Vec::with_capacity(model.forward_transitions.len());

    for t in &model.forward_transitions {
        if let Some(g) = g_evaluators.remove(&t.get_full_name()) {
            g_evaluator_vec.push(create_g_evaluator(g));
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "No g evaluator found for transition {}",
                t.get_full_name()
            )));
        }
    }

    Ok((forced_g_evaluator_vec, g_evaluator_vec))
}

pub fn create_h_evaluator(
    h_evaluator: Py<PyAny>,
) -> impl Fn(&StateInRegistry) -> Option<OrderedContinuous> {
    move |state: &_| {
        Python::with_gil(|py| {
            let state = State::from(state.clone());
            let args = PyTuple::new(py, &[StatePy::from(state).into_py(py)]);
            h_evaluator
                .call1(py, args)
                .unwrap()
                .extract::<Option<Continuous>>(py)
                .unwrap()
                .map(OrderedContinuous::from)
        })
    }
}

pub fn create_policy(policy: Py<PyAny>) -> impl Fn(&StateInRegistry) -> Vec<f64> {
    move |state: &_| {
        Python::with_gil(|py| {
            let state = State::from(state.clone());
            let args = PyTuple::new(py, &[StatePy::from(state).into_py(py)]);
            policy
                .call1(py, args)
                .unwrap()
                .extract::<Vec<f64>>(py)
                .unwrap()
        })
    }
}

pub fn create_policy_f_evaluator<T>(evaluator: Py<PyAny>) -> impl Fn(f64, T, Option<T>) -> f64
where
    T: Numeric,
{
    move |pi, g, h| {
        Python::with_gil(|py| {
            let args = PyTuple::new(
                py,
                &[
                    pi.into_py(py),
                    g.to_continuous().into_py(py),
                    h.map(|h| h.to_continuous()).into_py(py),
                ],
            );
            evaluator
                .call1(py, args)
                .unwrap()
                .extract::<f64>(py)
                .unwrap()
        })
    }
}
