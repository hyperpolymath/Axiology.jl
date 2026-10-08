# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

"""
    Axiology

Value theory for machine-learning and decision systems: make the values a
system is meant to serve explicit, typed and checkable.

Axiology (from Greek ἀξία, *axía*, "value", and -λογία, *-logia*, "study") is the
philosophical study of value. This package gives a small set of value concepts a
computational form so they can be stated, scored and compared:

1. **Value types** — `Fairness`, `Welfare`, `Profit`, `Efficiency` and `Safety`,
   all subtypes of `Value`, each carrying its metric, threshold or target, and a
   weight.
2. **Satisfaction checks** — `satisfy(value, state)` reports whether a system
   state (a `Dict` of predictions, utilities, profit, timings, …) meets the value.
3. **Fairness and welfare measures** — `demographic_parity`, `equalized_odds`,
   `equal_opportunity`, `disparate_impact`, `individual_fairness`;
   `utilitarian_welfare`, `rawlsian_welfare`, `egalitarian_welfare`.
4. **Multi-objective comparison** — `value_score`, `weighted_score`,
   `normalize_scores`, `dominated` and `pareto_frontier` compare candidate
   states across several values without hiding the trade-offs.
5. **Verification attestations** — `verify_value` checks the shape of a
   caller-supplied proof attestation. It does not run a prover.

# Example

```julia
using Axiology

fairness = Fairness(metric = :demographic_parity,
                    protected_attributes = [:group],
                    threshold = 0.05)

state = Dict(
    :predictions => [1, 0, 1, 1, 0, 1],
    :protected   => [:a, :a, :b, :b, :b, :a],
)

satisfy(fairness, state)  # true: both groups have a positive rate of 2/3

values = [
    Welfare(metric = :utilitarian, weight = 0.5),
    Fairness(metric = :demographic_parity, threshold = 0.1, weight = 0.5),
]
candidates = [
    Dict(:utilities => [3.0, 3.0], :predictions => [1, 1], :protected => [:a, :b]),
    Dict(:utilities => [9.0, 0.0], :predictions => [1, 0], :protected => [:a, :b]),
]
pareto_frontier(candidates, values)  # both survive: one is fairer, the other has more welfare
```
"""
module Axiology

using Statistics

# Core value types
export Value, Fairness, Welfare, Profit, Efficiency, Safety
export FairnessMetric, WelfareMetric, EfficiencyMetric
# Enum variant exports
export demographic_parity_metric, equalized_odds_metric, equal_opportunity_metric
export disparate_impact_metric, individual_fairness_metric
export utilitarian_metric, rawlsian_metric, egalitarian_metric
export pareto_metric, kaldor_hicks_metric, computation_time_metric
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
