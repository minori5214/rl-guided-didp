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
    create_policy_guided_dual_bound_acps, create_policy_guided_user_priority_acps,
    create_user_priority_acps, FEvaluatorType, Parameters, PolicyEvaluators,
    ProgressiveSearchParameters, Search, UserEvaluators,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

/// Anytime Column Progressive Seach (ACPS) solver.
///
/// This performs ACPS using user-provided g-, h-, and f-values and a user-provided policy.
/// The state **minimizes** the f- and h- values has the highest priority.
/// When `h_evaluator` is not provided, `policy` must be provided, the dual bound function defined in the model is used as the heuristic function,
/// and the f-value is computed by `policy_priority_evaluator`, which combines the cost, h-value, and the log probability given by the policy.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
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
///     Primal bound on the optimal cost (upper/lower bound for minimization/maximization).
/// time_limit: int, float, or None, default: None
///     Time limit.
/// get_all_solutions: bool, default: False
///     Return a new solution even if it is not improving when :code:`search_next()` is called.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     Initial size of the data structure storing all generated states.
/// width_init: int, default: 1
///     Initial value of the width.
/// width_step: int, default: 1
///     Amount of increase of the width.
/// width_bound: int or None, default: None
///     Maximum value of the width.
/// reset_width: bool, default: False
///     Reset the width to :code:`width_init` when a solution is found.
/// expansion_limit: int or None, default: None
///     Expansion limit.
///
/// Raises
/// ------
/// TypeError
///     If :code:`primal_bound` is :code:`float` and :code:`model` is int cost.
/// OverflowError
///     If :code:`initial_registry_capacity` is negative.
/// PanicException
///     If :code:`time_limit` is negative.
///
/// References
/// ----------
/// Ryo Kuroiwa and J. Christopher Beck.
/// "Domain-Independent Dynamic Programming: Generic State Space Search for Combinatorial Optimization,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 236-244, 2023.
///
/// Stephen Edelkamp, Shahid Jabbar, Alberto Lluch Lafuente.
/// "Cost-Algebraic Heuristic Search,"
/// Proceedings of the 20th National Conference on Artificial Intelligence (AAAI), pp. 1362-1367, 2005.
///
/// Peter E. Hart, Nills J. Nilsson, Bertram Raphael.
/// "A Formal Basis for the Heuristic Determination of Minimum Cost Paths",
/// IEEE Transactions of Systems Science and Cybernetics, vol. SSC-4(2), pp. 100-107, 1968.
#[pyclass(unsendable, name = "UserPriorityACPS")]
pub struct UserPriorityAcpsPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

#[pymethods]
impl UserPriorityAcpsPy {
    #[new]
    #[pyo3(
        text_signature = "(model, h_evaluator=None, g_evaluators=None, f_operator=None, policy=None, policy_f_evaluator=None, no_policy_accumulation=False, bound_operator=didppy.FOperator.Plus, primal_bound=None, time_limit=None, get_all_solutions=False, quiet=False, initial_registry_capacity=1000000, width_init=1, width_step=1, width_bound=None, reset_width=False)"
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
        get_all_solutions = false,
        quiet = false,
        initial_registry_capacity = 1000000,
        width_init = 1,
        width_step = 1,
        width_bound = None,
        reset_width = false,
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
        get_all_solutions: bool,
        quiet: bool,
        initial_registry_capacity: usize,
        width_init: usize,
        width_step: usize,
        width_bound: Option<usize>,
        reset_width: bool,
        expansion_limit: Option<usize>,
    ) -> PyResult<UserPriorityAcpsPy> {
        if !quiet {
            println!(
                "Solver: UserPriorityACPS from DIDPPy v{}",
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

        let progressive_parameters = ProgressiveSearchParameters {
            init: width_init,
            step: width_step,
            bound: width_bound,
            reset: reset_width,
        };

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
                let parameters = Parameters::<OrderedContinuous> {
                    primal_bound,
                    time_limit,
                    get_all_solutions,
                    quiet,
                    initial_registry_capacity: Some(initial_registry_capacity),
                    expansion_limit,
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
                    create_policy_guided_user_priority_acps(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                        policy_evaluators,
                        progressive_parameters,
                    )
                } else {
                    create_user_priority_acps(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                        progressive_parameters,
                    )
                };

                Ok(UserPriorityAcpsPy(WrappedSolver::Float(solver)))
            } else {
                let primal_bound = if let Some(primal_bound) = primal_bound {
                    Some(primal_bound.extract::<Integer>()?)
                } else {
                    None
                };
                let parameters = Parameters::<Integer> {
                    primal_bound,
                    time_limit,
                    get_all_solutions,
                    quiet,
                    initial_registry_capacity: Some(initial_registry_capacity),
                    expansion_limit,
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
                    create_policy_guided_user_priority_acps(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                        policy_evaluators,
                        progressive_parameters,
                    )
                } else {
                    create_user_priority_acps(
                        model,
                        parameters,
                        bound_evaluator_type,
                        user_evaluators,
                        progressive_parameters,
                    )
                };

                Ok(UserPriorityAcpsPy(WrappedSolver::Int(solver)))
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
                let parameters = Parameters::<OrderedContinuous> {
                    primal_bound,
                    time_limit,
                    get_all_solutions,
                    quiet,
                    initial_registry_capacity: Some(initial_registry_capacity),
                    expansion_limit,
                };
                let policy_evaluators = PolicyEvaluators {
                    policy: create_policy(policy),
                    f_evaluator: create_policy_f_evaluator(priority_evaluator),
                    no_accumulation: no_policy_accumulation,
                };
                let solver = create_policy_guided_dual_bound_acps(
                    model,
                    parameters,
                    bound_evaluator_type,
                    policy_evaluators,
                    progressive_parameters,
                );
                Ok(UserPriorityAcpsPy(WrappedSolver::Float(solver)))
            } else {
                let primal_bound = if let Some(primal_bound) = primal_bound {
                    Some(primal_bound.extract::<Integer>()?)
                } else {
                    None
                };
                let parameters = Parameters::<Integer> {
                    primal_bound,
                    time_limit,
                    get_all_solutions,
                    quiet,
                    initial_registry_capacity: Some(initial_registry_capacity),
                    expansion_limit,
                };
                let policy_evaluators = PolicyEvaluators {
                    policy: create_policy(policy),
                    f_evaluator: create_policy_f_evaluator(priority_evaluator),
                    no_accumulation: no_policy_accumulation,
                };
                let solver = create_policy_guided_dual_bound_acps(
                    model,
                    parameters,
                    bound_evaluator_type,
                    policy_evaluators,
                    progressive_parameters,
                );
                Ok(UserPriorityAcpsPy(WrappedSolver::Int(solver)))
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
    /// >>> solver = dp.UserPriorityACPS(model, quiet=True)
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
    /// >>> solver = dp.UserPriorityACPS(model, quiet=True)
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
