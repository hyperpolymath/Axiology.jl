# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Core value type system for ethical and economic values in ML.
"""

# Abstract base type
"""
    Value

Abstract base type for all value types in Axiology.

Values represent ethical or economic objectives that ML systems should optimize or satisfy.
"""
abstract type Value end

# Fairness metrics enum
@enum FairnessMetric begin
    demographic_parity_metric
    equalized_odds_metric
    equal_opportunity_metric
    disparate_impact_metric
    individual_fairness_metric
end

# Welfare metrics enum
@enum WelfareMetric begin
    utilitarian_metric
    rawlsian_metric
    egalitarian_metric
end

# Efficiency metrics enum
@enum EfficiencyMetric begin
    pareto_metric
    kaldor_hicks_metric
    computation_time_metric
end

"""
    Fairness <: Value

Represents fairness constraints in ML models.

# Fields
- `metric::Symbol`: Fairness metric to use (`:demographic_parity`, `:equalized_odds`, etc.)
- `protected_attributes::Vector{Symbol}`: Attributes that should not affect predictions
- `threshold::Float64`: Maximum acceptable disparity (default: 0.05)
- `weight::Float64`: Weight in multi-objective optimization (default: 1.0)

# Example

```julia
fairness = Fairness(
    metric = :demographic_parity,
    protected_attributes = [:gender, :race],
    threshold = 0.05
)
```
"""
struct Fairness <: Value
    metric::Symbol
    protected_attributes::Vector{Symbol}
    threshold::Float64
    weight::Float64

    function Fairness(; metric::Symbol = :demographic_parity,
                        protected_attributes::Vector{Symbol} = Symbol[],
                        threshold::Float64 = 0.05,
                        weight::Float64 = 1.0)
        @assert metric in [:demographic_parity, :equalized_odds, :equal_opportunity,
                          :disparate_impact, :individual_fairness] "Invalid fairness metric"
        @assert threshold >= 0.0 && threshold <= 1.0 "Threshold must be in [0,1]"
        @assert weight >= 0.0 "Weight must be non-negative"
        new(metric, protected_attributes, threshold, weight)
    end
end

"""
    Welfare <: Value

Represents social welfare functions.

# Fields
- `metric::Symbol`: Welfare metric (`:utilitarian`, `:rawlsian`, `:egalitarian`)
- `weight::Float64`: Weight in multi-objective optimization (default: 1.0)

# Example

```julia
welfare = Welfare(metric = :utilitarian, weight = 0.5)
```
"""
struct Welfare <: Value
    metric::Symbol
    weight::Float64

    function Welfare(; metric::Symbol = :utilitarian, weight::Float64 = 1.0)
        @assert metric in [:utilitarian, :rawlsian, :egalitarian] "Invalid welfare metric"
        @assert weight >= 0.0 "Weight must be non-negative"
        new(metric, weight)
    end
end

"""
    Profit <: Value

Represents economic profit optimization.

# Fields
- `target::Float64`: Target profit value
- `constraints::Vector{Value}`: Additional value constraints
- `weight::Float64`: Weight in multi-objective optimization (default: 1.0)

# Example

```julia
profit = Profit(target = 1000000.0, constraints = [fairness, safety])
```
"""
struct Profit <: Value
    target::Float64
    constraints::Vector{Value}
    weight::Float64

    function Profit(; target::Float64 = 0.0,
                      constraints::Vector{Value} = Value[],
                      weight::Float64 = 1.0)
        @assert weight >= 0.0 "Weight must be non-negative"
        new(target, constraints, weight)
    end
end

"""
    Efficiency <: Value

Represents efficiency metrics (computational or economic).

# Fields
- `metric::Symbol`: Efficiency metric (`:pareto`, `:kaldor_hicks`, `:computation_time`)
- `target::Float64`: Target efficiency value
- `weight::Float64`: Weight in multi-objective optimization (default: 1.0)

# Example

```julia
efficiency = Efficiency(metric = :computation_time, target = 0.1)
```
"""
struct Efficiency <: Value
    metric::Symbol
    target::Float64
    weight::Float64

    function Efficiency(; metric::Symbol = :pareto,
                         target::Float64 = 1.0,
                         weight::Float64 = 1.0)
        @assert metric in [:pareto, :kaldor_hicks, :computation_time] "Invalid efficiency metric"
        @assert weight >= 0.0 "Weight must be non-negative"
        new(metric, target, weight)
    end
end

"""
    Safety <: Value

Represents safety invariants and constraints.

# Fields
- `invariant::String`: Safety invariant (logical formula or description)
- `critical::Bool`: Whether this is a critical safety constraint
- `weight::Float64`: Weight in multi-objective optimization (default: 1.0)

# Example

```julia
safety = Safety(
    invariant = "No harmful recommendations",
    critical = true
)
```
"""
struct Safety <: Value
    invariant::String
    critical::Bool
    weight::Float64

    function Safety(; invariant::String = "",
                     critical::Bool = true,
                     weight::Float64 = 1.0)
        @assert !isempty(invariant) "Safety invariant cannot be empty"
        @assert weight >= 0.0 "Weight must be non-negative"
        new(invariant, critical, weight)
    end
end

# Display methods
Base.show(io::IO, f::Fairness) = print(io, "Fairness(:$(f.metric), threshold=$(f.threshold))")
Base.show(io::IO, w::Welfare) = print(io, "Welfare(:$(w.metric))")
Base.show(io::IO, p::Profit) = print(io, "Profit(target=$(p.target))")
Base.show(io::IO, e::Efficiency) = print(io, "Efficiency(:$(e.metric), target=$(e.target))")
Base.show(io::IO, s::Safety) = print(io, "Safety(\"$(s.invariant)\")")
