# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
    Axiology

Value theory for machine learning - define, optimize, and verify ethical/economic values in ML models.

This module provides:
- Value type system (Fairness, Welfare, Profit, Efficiency, Safety)
- Value satisfaction checking
- Multi-objective optimization
- Pareto frontier analysis
- Value verification

# Example

```julia
using Axiology

# Define fairness criterion
fairness = Fairness(
    metric = :demographic_parity,
    protected_attributes = [:gender, :race],
    threshold = 0.05
)

# Check if a model satisfies fairness
state = Dict(
    :predictions => [0.8, 0.7, 0.6, 0.9],
    :protected => [:male, :female, :male, :female],
    :disparity => 0.03
)

@assert satisfy(fairness, state)  # Disparity < threshold

# Multi-objective optimization
values = [
    Welfare(metric = :utilitarian, weight = 0.4),
    Fairness(metric = :demographic_parity, weight = 0.3),
    Efficiency(metric = :computation_time, weight = 0.3)
]

solutions = pareto_frontier(system, values)
```
"""
module Axiology

using Statistics
using LinearAlgebra

# Core value types
export Value, Fairness, Welfare, Profit, Efficiency, Safety
export FairnessMetric, WelfareMetric, EfficiencyMetric
export satisfy, maximize, verify_value
export pareto_frontier, dominated, value_score
export weighted_score, normalize_scores

# Fairness metrics
export demographic_parity, equalized_odds, equal_opportunity
export disparate_impact, individual_fairness

# Welfare functions
export utilitarian_welfare, rawlsian_welfare, egalitarian_welfare

# Value types and implementations
include("types.jl")
include("fairness.jl")
include("welfare.jl")
include("optimization.jl")

end # module Axiology
