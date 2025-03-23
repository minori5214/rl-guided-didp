use super::f_operator::FOperator;
use super::user_priority_evaluator::{
    create_default_g_evaluator_vectors, create_g_evaluator_vectors, create_h_evaluator,
    create_policy, create_policy_f_evaluator,
};
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_policy_guided_dual_bound_cabs, create_policy_guided_user_priority_cabs,
    create_user_priority_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters,
    PolicyEvaluators, Search, UserEvaluators,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

/// Complete Anytime Beam Search (CABS) solver.
///
/// This performs CABS using user-provided g-, h-, and f-values.
/// The state **minimizes** the f- and h- values has the highest priority.
/// When `h_evaluator` is not provided, `policy` must be provided, the dual bound function defined in the model is used as the heuristic function,
/// and the f-value is computed by `policy_priority_evaluator`, which combines the cost, h-value, and the log probability given by the policy.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
///
/// CABS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If :code:`keep_all_layers` is :code:`True`, CABS keeps states in all layers to check for duplicates.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// h_evaluator: Callable[[State], float] or None, default: None
///     Heuristic function.
/// g_evaluators: Dict[str, Callable[[float, State], float]] or None, default: None
///     Dictionary of g-value evaluators.
///     If `None`, the g-value is computed by the cost expression of the transition.
/// f_operator: FOperator or None, default: None
///     Operator to combine a g-value and an h-value to compute an f-value.
///     If `None`, `bound_operator` is used.
///     When `policy_priority_evaluator` is provided, the f-value is overwritten by `policy_priority_evaluator`.
///     However, the value computed by `f_operator` is still used to break ties.
/// policy: Callable[[State], List[float]] or None, default: None
///     Function to compute a probability distribution over transitions.
///     It should return a list of the **log probabilities** for **non-forced** transitions.
///     The order should match the order in which the transitions are defined in the model.
///     The log probabilities of unapplicable transitions do not matter.
/// policy_f_evaluator: Callable[[float, float, float], float] or None, default: None
///     Function to compute the f-value using the g-value, h-value, and the log probability given by the policy.
///     Note that when `h_evaluator` is not provided and any dual bound function is defined in the model, the g-value is `None`.
/// no_policy_accumulation: bool, default: False
///     Whether to accumulate the log probabilities or not.
///     If `True`, then the log probability at each state is the log probability of the transition from the parent state to the state.
///     If `False`, then the log probability at each state is computed by summing the log probabilities of the transitions from the root state to the state.
/// bound_operator: FOperator, default: FOperator.Plus
///     Operator to combine the cost and the return value of the dual bound function to compute the dual bound.
///     If the cost is computed by :code:`+`, this should be :attr:`~FOperator.Plus`.
///     If the cost is computed by :code:`*`, this should be :attr:`~FOperator.Product`.
///     If the cost is computed by :code:`max`, this should be :attr:`~FOperator.Max`.
///     If the cost is computed by :code:`min`, this should be :attr:`~FOperator.Min`.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int, float, or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_beam_size: int, default: 1
///     Initial beam size.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// max_beam_size: int or None, default: None
///     Maximum beam size.
///     If `None`, the beam size is kept increased until proving optimality or infeasibility or reaching the time limit.
/// expansion_limit: int or None, default: None
///     Expansion limit.
///
/// Raises
/// ------
/// TypeError
///     If :code:`primal_bound` is :code:`float` and :code:`model` is int cost.
/// PanicException
///     If :code:`time_limit` is negative.
///
/// References
/// ----------
/// Ryo Kuroiwa and J. Christopher Beck.
/// "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// Weixiong Zhang.
/// "Complete Anytime Beam Search,"
/// Proceedings of the 15th National Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence (AAAI/IAAI), pp. 425-430, 1998.
#[pyclass(unsendable, name = "UserPriorityCABS")]
pub struct UserPriorityCabsPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

#[pymethods]
impl UserPriorityCabsPy {
    #[new]
    #[pyo3(
        text_signature = "(model, h_evaluator=None, g_evaluator=None, f_operator=None, policy=None, policy_f_evaluator=None, policy_no_accumulation=False, bound_operator=didppy.FOperator.Plus, primal_bound=None, time_limit=None, quiet=False, initial_beam_size=1, keep_all_layers=False, max_beam_size=None)"
    )]
    #[pyo3(signature = (
        model,
        h_evaluator = None,
        g_evaluators = None,
        f_operator = None,
        policy = None,
        policy_f_evaluator = None,
        no_policy_accumulation = false,
        bound_operator = FOperator::Plus,
        primal_bound = None,
        time_limit = None,
        quiet = false,
        initial_beam_size = 1,
        keep_all_layers = false,
        max_beam_size = None,
        expansion_limit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        h_evaluator: Option<Py<PyAny>>,
        g_evaluators: Option<HashMap<String, Py<PyAny>>>,
        f_operator: Option<FOperator>,
        policy: Option<Py<PyAny>>,
        policy_f_evaluator: Option<Py<PyAny>>,
        no_policy_accumulation: bool,
        bound_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_beam_size: usize,
        keep_all_layers: bool,
        max_beam_size: Option<usize>,
        expansion_limit: Option<usize>,
    ) -> PyResult<UserPriorityCabsPy> {
        if !quiet {
            println!(
                "Solver: UserPriorityCABS from DIDPPy v{}",
                env!("CARGO_PKG_VERSION")
            );
        }

        if h_evaluator.is_none() && policy.is_none() {
            return Err(PyRuntimeError::new_err(
                "`h_evaluator` or `policy` must be provided",
            ));
        }

        if policy.is_some() != policy_f_evaluator.is_some() {
            return Err(PyRuntimeError::new_err(
                "`policy` and `policy_priority_evaluator` must be both provided or both not provided",
            ));
        }

        let bound_evaluator_type = FEvaluatorType::from(bound_operator.clone());

        let float_cost = model.float_cost();
        let model = Rc::new(model.inner_as_ref().clone());

        if let Some(h_evaluator) = h_evaluator {
            let (forced_g_evaluators, g_evaluators) = if let Some(mut g_evaluators) = g_evaluators {
                create_g_evaluator_vectors(&model, &mut g_evaluators)?
            } else {
                create_default_g_evaluator_vectors(&model)
            };
            let h_evaluator = create_h_evaluator(h_evaluator);
            let f_evaluator_type = FEvaluatorType::from(f_operator.unwrap_or(bound_operator));

            if float_cost {
                let primal_bound = if let Some(primal_bound) = primal_bound {
                    Some(OrderedContinuous::from(
                        primal_bound.extract::<Continuous>()?,
                    ))
                } else {
                    None
                };
                let parameters = CabsParameters {
                    max_beam_size,
                    beam_search_parameters: BeamSearchParameters {
                        beam_size: initial_beam_size,
                        keep_all_layers,
                        parameters: Parameters::<OrderedContinuous> {
                            primal_bound,
                            time_limit,
                            get_all_solutions: false,
                            quiet,
                            initial_registry_capacity: None,
                            expansion_limit,
                        },
                    },
                };
                let user_evaluators = UserEvaluators {
                    forced_g_evaluators,
                    g_evaluators,
                    h_evaluator,
                    f_evaluator_type,
                };

                let solver = if let (Some(policy), Some(priority_evaluator)) =
                    (policy, policy_f_evaluator)
                {
                    let policy_evaluators = PolicyEvaluators {
                        policy: create_policy(policy),
                        f_evaluator: create_policy_f_evaluator(priority_evaluator),
                        no_accumulation: no_policy_accumulation,
                    };
                    create_policy_guided_user_priority_cabs(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                        policy_evaluators,
                    )
                } else {
                    create_user_priority_cabs(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                    )
                };

                Ok(UserPriorityCabsPy(WrappedSolver::Float(solver)))
            } else {
                let primal_bound = if let Some(primal_bound) = primal_bound {
                    Some(primal_bound.extract::<Integer>()?)
                } else {
                    None
                };
                let parameters = CabsParameters {
                    max_beam_size,
                    beam_search_parameters: BeamSearchParameters {
                        beam_size: initial_beam_size,
                        keep_all_layers,
                        parameters: Parameters::<Integer> {
                            primal_bound,
                            time_limit,
                            get_all_solutions: false,
                            quiet,
                            initial_registry_capacity: None,
                            expansion_limit,
                        },
                    },
                };
                let user_evaluators = UserEvaluators {
                    forced_g_evaluators,
                    g_evaluators,
                    h_evaluator,
                    f_evaluator_type,
                };

                let solver = if let (Some(policy), Some(priority_evaluator)) =
                    (policy, policy_f_evaluator)
                {
                    let policy_evaluators = PolicyEvaluators {
                        policy: create_policy(policy),
                        f_evaluator: create_policy_f_evaluator(priority_evaluator),
                        no_accumulation: no_policy_accumulation,
                    };
                    create_policy_guided_user_priority_cabs(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                        policy_evaluators,
                    )
                } else {
                    create_user_priority_cabs(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                    )
                };

                Ok(UserPriorityCabsPy(WrappedSolver::Int(solver)))
            }
        } else {
            let policy = policy.unwrap();
            let priority_evaluator = policy_f_evaluator.unwrap();

            if float_cost {
                let primal_bound = if let Some(primal_bound) = primal_bound {
                    Some(OrderedContinuous::from(
                        primal_bound.extract::<Continuous>()?,
                    ))
                } else {
                    None
                };
                let parameters = CabsParameters {
                    max_beam_size,
                    beam_search_parameters: BeamSearchParameters {
                        beam_size: initial_beam_size,
                        keep_all_layers,
                        parameters: Parameters::<OrderedContinuous> {
                            primal_bound,
                            time_limit,
                            get_all_solutions: false,
                            quiet,
                            initial_registry_capacity: None,
                            expansion_limit,
                        },
                    },
                };
                let policy_evaluators = PolicyEvaluators {
                    policy: create_policy(policy),
                    f_evaluator: create_policy_f_evaluator(priority_evaluator),
                    no_accumulation: no_policy_accumulation,
                };
                let solver = create_policy_guided_dual_bound_cabs(
                    model,
                    parameters,
                    bound_evaluator_type,
                    policy_evaluators,
                );
                Ok(UserPriorityCabsPy(WrappedSolver::Float(solver)))
            } else {
                let primal_bound = if let Some(primal_bound) = primal_bound {
                    Some(primal_bound.extract::<Integer>()?)
                } else {
                    None
                };
                let parameters = CabsParameters {
                    max_beam_size,
                    beam_search_parameters: BeamSearchParameters {
                        beam_size: initial_beam_size,
                        keep_all_layers,
                        parameters: Parameters::<Integer> {
                            primal_bound,
                            time_limit,
                            get_all_solutions: false,
                            quiet,
                            initial_registry_capacity: None,
                            expansion_limit,
                        },
                    },
                };
                let policy_evaluators = PolicyEvaluators {
                    policy: create_policy(policy),
                    f_evaluator: create_policy_f_evaluator(priority_evaluator),
                    no_accumulation: no_policy_accumulation,
                };
                let solver = create_policy_guided_dual_bound_cabs(
                    model,
                    parameters,
                    bound_evaluator_type,
                    policy_evaluators,
                );
                Ok(UserPriorityCabsPy(WrappedSolver::Int(solver)))
            }
        }
    }

    /// search()
    ///
    /// Search for the optimal solution of a DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the model is invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> x = model.add_int_var(target=1)
    /// >>> model.add_base_case([x == 0])
    /// >>> t = dp.Transition(
    /// ...     name="decrement",
    /// ...     cost=1 + dp.IntExpr.state_cost(),
    /// ...     effects=[(x, x - 1)]
    /// ... )
    /// >>> model.add_transition(t)
    /// >>> model.add_dual_bound(x)
    /// >>> solver = dp.CABS(model, quiet=True)
    /// >>> solution = solver.search()
    /// >>> solution.cost
    /// 1
    #[pyo3(signature = ())]
    fn search(&mut self) -> PyResult<SolutionPy> {
        self.0.search()
    }

    /// search_next()
    ///
    /// Search for the next solution of a DyPDL model.
    ///
    /// Returns
    /// -------
    /// solution: Solution
    ///     Solution.
    /// terminated: bool
    ///     Whether the search is terminated.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the model is invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> x = model.add_int_var(target=1)
    /// >>> model.add_base_case([x == 0])
    /// >>> t = dp.Transition(
    /// ...     name="decrement",
    /// ...     cost=1 + dp.IntExpr.state_cost(),
    /// ...     effects=[(x, x - 1)]
    /// ... )
    /// >>> model.add_transition(t)
    /// >>> model.add_dual_bound(x)
    /// >>> solver = dp.CABS(model, quiet=True)
    /// >>> solution, terminated = solver.search_next()
    /// >>> solution.cost
    /// 1
    /// >>> terminated
    /// True
    #[pyo3(signature = ())]
    fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        self.0.search_next()
    }
}
