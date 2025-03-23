use crate::FEvaluatorType;

/// User-defined evaluators.
pub struct UserEvaluators<G, H> {
    /// Evaluators for g-values for forced transitions.
    pub forced_g_evaluators: Vec<G>,
    /// Evaluators for g-values for transitions.
    pub g_evaluators: Vec<G>,
    /// Evaluators for h-values.
    pub h_evaluator: H,
    /// Type of f-evaluator.
    pub f_evaluator_type: FEvaluatorType,
}

/// Policy evaluators.
pub struct PolicyEvaluators<P, F> {
    /// Policy that should return the log probabilities of non-forced transitions.
    pub policy: P,
    /// Function that computes f-value from the log probability, g-value and h-value.
    pub f_evaluator: F,
    /// Whether to accumulate log probabilities along a path.
    pub no_accumulation: bool,
}
