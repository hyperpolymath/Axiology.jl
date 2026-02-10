# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Social Welfare Function Implementations and Value Satisfaction/Optimization Methods

This file contains implementations of various social welfare functions, which are
used to aggregate individual utilities or outcomes into a single measure of collective
well-being. These functions are crucial for evaluating and optimizing machine
learning systems that impact multiple individuals, especially when considering
ethical objectives beyond simple aggregate performance.

The different welfare functions reflect distinct philosophical approaches to
distributive justice:
-   **Utilitarianism**: Focuses on maximizing the sum of individual utilities.
-   **Rawlsianism (Maximin)**: Prioritizes improving the well-being of the worst-off.
-   **Egalitarianism**: Aims to reduce inequality and promote more equal distributions
    of well-being.

In addition to welfare functions, this file also provides overloaded `satisfy`,
`maximize`, and `verify_value` methods for various `Value` types (`Welfare`, `Profit`,
`Efficiency`, `Safety`). These methods operationalize the checking and optimization
of these values within an ML context.
"""

"""
    utilitarian_welfare(utilities::AbstractVector{<:Real})::Float64

Computes the utilitarian social welfare score for a given set of individual
utility values. Utilitarianism is a consequentialist ethical framework that
advocates for actions or policies that maximize overall happiness or well-being.
In the context of ML, this typically means optimizing the model to produce
outcomes that yield the highest sum of utility across all affected individuals.

The utilitarian welfare `W` is calculated as the sum of all individual utility values:
`W = Σᵢ uᵢ`

# Arguments
- `utilities::AbstractVector{<:Real}`: A vector where each element represents the
                                       utility or well-being experienced by an
                                       individual. These values can be positive,
                                       negative, or zero, and their scale should
                                       be consistent.

# Returns
- `Float64`: The total utilitarian welfare score, which is the sum of all
             individual utilities.

# Implications and Critiques in ML:
-   **Maximizes Aggregate Benefit**: Directly optimizes for the greatest good for the
    greatest number, which can be desirable for system-level efficiency.
-   **Potential for Inequality**: A major critique is that it can justify outcomes
    where some individuals experience very low utility if it leads to a sufficiently
    high aggregate sum. It does not inherently guarantee fair distribution.
-   **Measurement Challenge**: Quantifying "utility" for ML outcomes (e.g., job
    recommendations, medical diagnoses) can be complex and subjective.

# Example

```julia
utilities = [10.0, 8.0, 12.0, 7.0] # Individual satisfaction scores
welfare = utilitarian_welfare(utilities)  # Returns 37.0

# Example with potential negative utilities
utilities_mixed = [5.0, -2.0, 10.0, 1.0]
welfare_mixed = utilitarian_welfare(utilities_mixed) # Returns 14.0
```
"""
function utilitarian_welfare(utilities::AbstractVector{<:Real})::Float64
    isempty(utilities) && return 0.0 # Return 0.0 for empty utility vector
    return sum(utilities)
end

"""
    rawlsian_welfare(utilities::AbstractVector{<:Real})::Float64

Computes the Rawlsian social welfare score for a given set of individual
utility values. Rawlsian theory, particularly John Rawls's "difference principle,"
advocates for maximizing the well-being of the worst-off individual or group
in society. In ML, this translates to designing models that prioritize
mitigating harm or improving outcomes for those who are most disadvantaged
by the system.

The Rawlsian welfare `W` is calculated as the minimum of all individual utility values:
`W = min(u₁, u₂, ..., uₙ)`

# Arguments
- `utilities::AbstractVector{<:Real}`: A vector where each element represents the
                                       utility or well-being experienced by an
                                       individual.

# Returns
- `Float64`: The Rawlsian welfare score, which is the minimum utility value
             found in the `utilities` vector.

# Implications and Critiques in ML:
-   **Prioritizes the Vulnerable**: Directly addresses concerns about protecting
    and uplifting the most vulnerable populations affected by ML systems.
-   **Potentially Suboptimal Aggregate**: A critique is that it might not maximize
    overall aggregate welfare if improving the worst-off requires significant
    sacrifices from others.
-   **Sensitivity to Outliers**: Highly sensitive to the lowest utility value,
    which can sometimes be an outlier or measurement error.

# Example

```julia
utilities = [10.0, 8.0, 12.0, 7.0]
welfare = rawlsian_welfare(utilities)  # Returns 7.0 (the worst-off)

# Example where prioritizing the worst-off changes the outcome
utilities_with_harm = [10.0, 8.0, 12.0, -100.0]
welfare_harm = rawlsian_welfare(utilities_with_harm) # Returns -100.0
```
"""
function rawlsian_welfare(utilities::AbstractVector{<:Real})::Float64
    isempty(utilities) && error("Cannot compute Rawlsian welfare for an empty utility vector.")
    return minimum(utilities)
end
    return minimum(utilities)
end

"""
    egalitarian_welfare(utilities::AbstractVector{<:Real})::Float64

Computes an egalitarian social welfare score based on the negative variance
of individual utility values. Egalitarianism, as a philosophical concept,
prioritizes equality and aims to reduce disparities among individuals. In ML,
this means designing models that lead to more equitable distributions of
outcomes or utilities across all affected parties.

This function uses negative variance as a proxy for equality:
`W = -Var(utilities)`
A lower variance implies greater equality, thus resulting in a higher (less
negative) welfare score. Perfect equality (zero variance) yields a welfare
score of `0.0`.

# Arguments
- `utilities::AbstractVector{<:Real}`: A vector where each element represents the
                                       utility or well-being experienced by an
                                       individual.

# Returns
- `Float64`: The negative of the variance of the `utilities` vector.
             - A value of `0.0` indicates perfect equality (all utilities are identical).
             - More negative values indicate greater inequality.

# Implications and Critiques in ML:
-   **Promotes Equality**: Directly optimizes for reducing disparities in outcomes,
    which is desirable for fairness and social cohesion.
-   **May Ignore Aggregate Levels**: A critique is that it primarily focuses on
    equality and might not consider the absolute levels of utility. For instance,
    two scenarios could have similar variance but vastly different average utilities.
-   **Data Requirements**: Requires at least two utility values to compute a meaningful
    variance.

# Example

```julia
utilities_equal = [10.0, 10.0, 10.0, 10.0]
welfare_equal = egalitarian_welfare(utilities_equal)  # Returns 0.0 (perfect equality)

utilities_unequal = [5.0, 10.0, 15.0, 20.0]
welfare_unequal = egalitarian_welfare(utilities_unequal)  # Returns -31.25 (higher inequality)
```
"""
function egalitarian_welfare(utilities::AbstractVector{<:Real})::Float64
    if length(utilities) < 2
        # Variance of a single element or empty set is undefined or 0 (which is equality).
        # For meaningful comparison, we might want to error or return 0.0
        # for zero variance for length 1.
        return 0.0 # No inequality if only one or zero individuals
    end
    return -var(utilities)
end
    return -var(utilities)
end

"""
    satisfy(value::Welfare, state::Dict)::Bool

Checks if a given system state satisfies the specified `Welfare` criterion's
minimum requirements. This function evaluates the current collective well-being
based on the chosen welfare metric and determines if it meets or exceeds a
defined minimum threshold.

# Arguments
- `value::Welfare`: A `Welfare` object specifying the metric (e.g., `:utilitarian`,
                    `:rawlsian`, `:egalitarian`).
- `state::Dict`: A dictionary representing the current system state.
                 It must contain:
    - `:utilities`: `AbstractVector{<:Real}` - A vector of individual utility values
                    from which the welfare score will be computed.
                 It can optionally contain:
    - `:min_welfare`: `Real` - The minimum acceptable welfare score. If not provided,
                      it defaults to `0.0`.

# Returns
- `Bool`: `true` if the computed welfare score is greater than or equal to
          the `:min_welfare` threshold, `false` otherwise.

# Throws
- `ErrorException`: If `:utilities` is missing from `state`.
- `ErrorException`: If an unknown welfare metric is specified in `value.metric`.

# Example

```julia
# Check if Rawlsian welfare (minimum utility) is above a threshold
rawlsian_welfare_obj = Welfare(metric = :rawlsian)
state_rawlsian = Dict(:utilities => [8.0, 9.0, 10.0], :min_welfare => 7.0)
println("Rawlsian Welfare Satisfied: $(satisfy(rawlsian_welfare_obj, state_rawlsian))") # true

# Check utilitarian welfare
utilitarian_welfare_obj = Welfare(metric = :utilitarian)
state_utilitarian = Dict(:utilities => [10.0, 10.0, 10.0], :min_welfare => 25.0)
println("Utilitarian Welfare Satisfied: $(satisfy(utilitarian_welfare_obj, state_utilitarian))") # true (30 >= 25)
```
"""
function satisfy(value::Welfare, state::Dict)::Bool
    utilities = get(state, :utilities, nothing)
    isnothing(utilities) && error("State must contain :utilities for Welfare satisfaction check.")

    min_welfare = get(state, :min_welfare, 0.0) # Default minimum welfare to 0.0 if not specified

    computed_welfare = if value.metric == :utilitarian
        utilitarian_welfare(utilities)
    elseif value.metric == :rawlsian
        rawlsian_welfare(utilities)
    elseif value.metric == :egalitarian
        egalitarian_welfare(utilities)
    else
        error("Unknown welfare metric: $(value.metric). Must be one of $(instances(WelfareMetric)).")
    end

    return computed_welfare >= min_welfare
end
    utilities = get(state, :utilities, nothing)
    isnothing(utilities) && error("State must contain :utilities")

    min_welfare = get(state, :min_welfare, 0.0)

    computed_welfare = if value.metric == :utilitarian
        utilitarian_welfare(utilities)
    elseif value.metric == :rawlsian
        rawlsian_welfare(utilities)
    elseif value.metric == :egalitarian
        egalitarian_welfare(utilities)
    else
        error("Unknown welfare metric: $(value.metric)")
    end

    return computed_welfare >= min_welfare
end

"""
    satisfy(value::Profit, state::Dict)::Bool

Checks if a given system state satisfies the specified `Profit` criterion.
This involves verifying two conditions:
1.  The `current_profit` in the `state` must meet or exceed the `value.target`.
2.  All additional `Value` constraints specified in `value.constraints` must also be satisfied.

This function allows for a holistic evaluation of profit objectives, ensuring that
economic goals are pursued within defined ethical or operational boundaries.

# Arguments
- `value::Profit`: A `Profit` object specifying the target profit and any associated constraints.
- `state::Dict`: A dictionary representing the current system state.
                 It must contain:
    - `:profit`: `Real` - The current profit achieved by the system.

# Returns
- `Bool`: `true` if the `current_profit` meets the `value.target` AND all
          `value.constraints` are satisfied in the given `state`, `false` otherwise.

# Throws
- `ErrorException`: If `:profit` is missing from `state`.
- Errors can also be thrown by `satisfy` methods for the constraints in `value.constraints`.

# Example

```julia
# Example with a simple profit target
profit_target = Profit(target = 1000.0)
state_high_profit = Dict(:profit => 1200.0)
state_low_profit = Dict(:profit => 800.0)
println("High profit state satisfied: $(satisfy(profit_target, state_high_profit))") # true
println("Low profit state satisfied: $(satisfy(profit_target, state_low_profit))")   # false

# Example with a profit target and a fairness constraint
fairness_constraint = Fairness(metric = :demographic_parity, threshold = 0.1)
profit_with_fairness = Profit(target = 5000.0, constraints = [fairness_constraint])

state_complex = Dict(
    :profit => 6000.0,
    :predictions => [1, 1, 0, 0, 1, 1],
    :protected => [:A, :B, :A, :B, :A, :B],
    :disparity => 0.05 # For demo, assume disparity is pre-calculated or extracted
)
println("Profit with fairness satisfied: $(satisfy(profit_with_fairness, state_complex))") # true if disparity is <= 0.1
```
"""
function satisfy(value::Profit, state::Dict)::Bool
    current_profit = get(state, :profit, nothing) # Use nothing to distinguish from actual 0.0 profit
    isnothing(current_profit) && error("State must contain :profit for Profit satisfaction check.")

    # Check profit target
    profit_ok = current_profit >= value.target

    # Check all constraints
    constraints_ok = all(satisfy(c, state) for c in value.constraints)

    return profit_ok && constraints_ok
end

"""
    satisfy(value::Efficiency, state::Dict)::Bool

Checks if a given system state satisfies the specified `Efficiency` criterion's
target. This function evaluates whether the system's performance, resource
utilization, or economic outcomes meet the defined efficiency standards.

# Arguments
- `value::Efficiency`: An `Efficiency` object specifying the metric (e.g.,
                       `:computation_time`, `:pareto`, `:kaldor_hicks`) and target.
- `state::Dict`: A dictionary representing the current system state or model evaluation
                 results. The required keys in this dictionary depend on the
                 `value.metric`:
    - `:computation_time`: Requires `:computation_time` (`Real`) - the actual time taken.
    - `:pareto`: Requires `:is_pareto_efficient` (`Bool`) - a flag indicating Pareto efficiency.
    - `:kaldor_hicks`: Requires `:net_gain` (`Real`) - the calculated net gain.

# Returns
- `Bool`: `true` if the system state satisfies the `Efficiency` criterion, `false` otherwise.

# Throws
- `ErrorException`: If a required key for a specific metric is missing from `state`.
- `ErrorException`: If an unknown efficiency metric is specified in `value.metric`.

# Example

```julia
# Check computational time efficiency
efficiency_time = Efficiency(metric = :computation_time, target = 0.5) # Target max 0.5s
state_time = Dict(:computation_time => 0.4)
println("Computational Efficiency Satisfied: $(satisfy(efficiency_time, state_time))") # true

# Check Pareto efficiency
efficiency_pareto = Efficiency(metric = :pareto)
state_pareto = Dict(:is_pareto_efficient => true)
println("Pareto Efficiency Satisfied: $(satisfy(efficiency_pareto, state_pareto))") # true

# Check Kaldor-Hicks efficiency (assuming target is minimum acceptable net gain)
efficiency_kh = Efficiency(metric = :kaldor_hicks, target = 100.0)
state_kh = Dict(:net_gain => 120.0)
println("Kaldor-Hicks Efficiency Satisfied: $(satisfy(efficiency_kh, state_kh))") # true
```
"""
function satisfy(value::Efficiency, state::Dict)::Bool
    if value.metric == :computation_time
        time = get(state, :computation_time, nothing)
        isnothing(time) && error("State must contain :computation_time for :computation_time efficiency check.")
        return time <= value.target
    elseif value.metric == :pareto
        is_pareto = get(state, :is_pareto_efficient, nothing)
        isnothing(is_pareto) && error("State must contain :is_pareto_efficient for :pareto efficiency check.")
        return is_pareto
    elseif value.metric == :kaldor_hicks
        net_gain = get(state, :net_gain, nothing)
        isnothing(net_gain) && error("State must contain :net_gain for :kaldor_hicks efficiency check.")
        return net_gain >= value.target
    else
        error("Unknown efficiency metric: $(value.metric). Must be one of $(instances(EfficiencyMetric)).")
    end
end

"""
    satisfy(value::Safety, state::Dict)::Bool

Checks if a given system state satisfies the specified `Safety` criterion.
This involves verifying that the system is generally considered safe and that
all defined safety invariants hold true within the current `state`.

# Arguments
- `value::Safety`: A `Safety` object specifying the safety invariant and criticality.
- `state::Dict`: A dictionary representing the current system state or evaluation.
                 It must contain:
    - `:is_safe`: `Bool` - A general indication of the system's safety status.
                         Defaults to `true` if not provided (optimistic assumption).
    - `:invariant_holds`: `Bool` - A flag indicating whether the specific
                                  `value.invariant` is currently satisfied.
                                  Defaults to `true` if not provided.

# Returns
- `Bool`: `true` if both `:is_safe` and `:invariant_holds` are `true` in the
          `state`, indicating that the safety criterion is met. `false` otherwise.

# Example

```julia
# A scenario where both general safety and specific invariant hold
safety_obj = Safety(invariant = "No data leakage")
state_safe = Dict(:is_safe => true, :invariant_holds => true)
println("Safe State 1 Satisfied: $(satisfy(safety_obj, state_safe))") # true

# A scenario where the invariant does not hold
state_unsafe_invariant = Dict(:is_safe => true, :invariant_holds => false)
println("Safe State 2 Satisfied: $(satisfy(safety_obj, state_unsafe_invariant))") # false

# A scenario where the system is generally not safe
state_unsafe_general = Dict(:is_safe => false, :invariant_holds => true)
println("Safe State 3 Satisfied: $(satisfy(safety_obj, state_unsafe_general))") # false
```
"""
function satisfy(value::Safety, state::Dict)::Bool
    # Check if safety invariant is satisfied in state
    is_safe = get(state, :is_safe, true) # Optimistic assumption if not provided
    invariant_holds = get(state, :invariant_holds, true) # Optimistic assumption if not provided

    return is_safe && invariant_holds
end

"""
    maximize(value::Welfare, initial_state::Dict)::Float64

Calculates the social welfare score for a given `Welfare` objective based on
the `initial_state`. This function provides a mechanism to quantify the
"goodness" of a system's outcome with respect to a chosen welfare function,
which can then be used in optimization processes.

# Arguments
- `value::Welfare`: A `Welfare` object specifying the metric (e.g., `:utilitarian`,
                    `:rawlsian`, `:egalitarian`).
- `initial_state::Dict`: A dictionary representing the current or candidate system state.
                         It must contain:
    - `:utilities`: `AbstractVector{<:Real}` - A vector of individual utility values
                    from which the welfare score will be computed.

# Returns
- `Float64`: The computed welfare score, reflecting the degree to which the
             `value.metric` is achieved in the `initial_state`.

# Throws
- `ErrorException`: If `:utilities` is missing from `initial_state`.
- `ErrorException`: If an unknown welfare metric is specified in `value.metric`.
- Errors can also be thrown by the underlying welfare functions (e.g., `rawlsian_welfare`
  if `:utilities` is empty).

# Example

```julia
# Maximize utilitarian welfare
utilitarian_obj = Welfare(metric = :utilitarian)
state_util = Dict(:utilities => [5.0, 6.0, 7.0])
score_util = maximize(utilitarian_obj, state_util) # Returns 18.0

# Maximize Rawlsian welfare
rawlsian_obj = Welfare(metric = :rawlsian)
state_rawlsian = Dict(:utilities => [5.0, 6.0, 7.0])
score_rawlsian = maximize(rawlsian_obj, state_rawlsian) # Returns 5.0
```
"""
function maximize(value::Welfare, initial_state::Dict)::Float64
    utilities = get(initial_state, :utilities, nothing)
    isnothing(utilities) && error("initial_state must contain :utilities for Welfare maximization.")

    if value.metric == :utilitarian
        return utilitarian_welfare(utilities)
    elseif value.metric == :rawlsian
        return rawlsian_welfare(utilities)
    elseif value.metric == :egalitarian
        return egalitarian_welfare(utilities)
    else
        error("Unknown welfare metric: $(value.metric). Must be one of $(instances(WelfareMetric)).")
    end
end

"""
    maximize(value::Profit, initial_state::Dict)::Float64

Calculates the profit score for a given `Profit` objective based on the
`initial_state`. This function is intended to provide a scalar value
that can be maximized during an optimization process to achieve the
desired profit goals.

# Arguments
- `value::Profit`: A `Profit` object specifying the target profit. Note that
                   while `Profit` objects can contain `constraints`, these
                   are handled by the `satisfy` method, and `maximize` here
                   simply returns the raw profit for single-objective scoring.
- `initial_state::Dict`: A dictionary representing the current or candidate system state.
                         It must contain:
    - `:profit`: `Real` - The current profit value achieved by the system.

# Returns
- `Float64`: The current profit value from `initial_state[:profit]`.

# Throws
- `ErrorException`: If `:profit` is missing from `initial_state`.

# Example

```julia
profit_obj = Profit(target = 1000.0)
state_profit = Dict(:profit => 1250.0)
score = maximize(profit_obj, state_profit) # Returns 1250.0
```
"""
function maximize(value::Profit, initial_state::Dict)::Float64
    profit = get(initial_state, :profit, nothing)
    isnothing(profit) && error("initial_state must contain :profit for Profit maximization.")
    return profit
end

"""
    maximize(value::Efficiency, initial_state::Dict)::Float64

Calculates an efficiency score for a given `Efficiency` objective based on the
`initial_state`. This function provides a scalar value that can be maximized
during an optimization process, typically by converting a minimization objective
(like `computation_time`) into a maximization one (by negating it).

# Arguments
- `value::Efficiency`: An `Efficiency` object specifying the metric and target.
- `initial_state::Dict`: A dictionary representing the current or candidate system state.
                         The required keys depend on `value.metric`:
    - `:computation_time`: Requires `:computation_time` (`Real`) - the actual time taken.
    - Other metrics (like `:pareto`, `:kaldor_hicks`) might require different keys or
      return a fixed score if they are typically handled as constraints rather than
      maximization objectives.

# Returns
- `Float64`: The computed efficiency score.
             - For `:computation_time`, it returns the negative of the time, so that
               minimizing time becomes maximizing `-time`.
             - For other metrics, it currently returns `1.0` as a placeholder,
               implying they are either not directly maximized here or treated
               as binary (satisfied/not satisfied).

# Throws
- `ErrorException`: If a required key for a specific metric is missing from `initial_state`.
- `ErrorException`: If an unknown efficiency metric is specified in `value.metric`.

# Example

```julia
# Maximize computational efficiency (minimize time)
efficiency_time_obj = Efficiency(metric = :computation_time)
state_time = Dict(:computation_time => 0.4) # 0.4 seconds
score_time = maximize(efficiency_time_obj, state_time) # Returns -0.4

state_longer_time = Dict(:computation_time => 1.0)
score_longer_time = maximize(efficiency_time_obj, state_longer_time) # Returns -1.0 (-0.4 > -1.0)
```
"""
function maximize(value::Efficiency, initial_state::Dict)::Float64
    if value.metric == :computation_time
        time = get(initial_state, :computation_time, nothing)
        isnothing(time) && error("initial_state must contain :computation_time for :computation_time efficiency maximization.")
        return -time  # Negative because we want to minimize time
    elseif value.metric == :pareto || value.metric == :kaldor_hicks
        # These are typically hard constraints or binary properties, not easily maximized
        # If they are satisfied, they contribute 1.0; otherwise 0.0.
        # This implementation currently just returns 1.0 as a neutral score
        # when not explicitly dealing with minimization (like time).
        return 1.0
    else
        error("Unknown efficiency metric: $(value.metric). Must be one of $(instances(EfficiencyMetric)).")
    end
end

"""
    maximize(value::Fairness, initial_state::Dict)::Float64

Calculates a fairness score for a given `Fairness` objective based on the
`initial_state`. This function provides a scalar value that can be maximized
during an optimization process, effectively by converting a minimization objective
(like disparity) into a maximization one (e.g., `1.0 - disparity`).

# Arguments
- `value::Fairness`: A `Fairness` object specifying the metric, protected attributes,
                     and threshold.
- `initial_state::Dict`: A dictionary representing the current or candidate system state.
                         It must contain:
    - `:predictions`: `AbstractVector` - Model predictions.
    - `:protected` or `:protected_attributes`: `AbstractVector` - Protected group indicators.
    - Additional keys might be required depending on `value.metric` (e.g., `:labels` for
      `equalized_odds`, `:similarity_matrix` for `individual_fairness`).

# Returns
- `Float64`: The computed fairness score. For metrics like `demographic_parity`,
             it returns `1.0 - disparity`, where `1.0` represents perfect fairness.
             Higher scores indicate better fairness.

# Throws
- `ErrorException`: If required keys for the specific `value.metric` are missing from `initial_state`.
- `ErrorException`: If an unknown fairness metric is specified in `value.metric`.

# Example

```julia
# Maximize demographic parity (minimize disparity)
fairness_obj = Fairness(metric = :demographic_parity)
state_fair = Dict(
    :predictions => [1, 0, 1, 1],
    :protected => [:male, :female, :male, :female]
)
score = maximize(fairness_obj, state_fair) # Returns 1.0 - demographic_parity(predictions, protected)
```
"""
function maximize(value::Fairness, initial_state::Dict)::Float64
    # Extract data from state
    predictions = get(initial_state, :predictions, nothing)
    protected = get(initial_state, :protected, get(initial_state, :protected_attributes, nothing))
    labels = get(initial_state, :labels, nothing)
    similarity_matrix = get(initial_state, :similarity_matrix, nothing)

    if isnothing(predictions) || (isnothing(protected) && value.metric != :individual_fairness) || (isnothing(similarity_matrix) && value.metric == :individual_fairness)
        # Cannot compute fairness score without required data
        return 0.0 # Return a low score to indicate poor fairness or inability to compute
    end

    # Compute disparity based on metric
    disparity_or_score = if value.metric == :demographic_parity
        demographic_parity(predictions, protected)
    elseif value.metric == :equalized_odds
        isnothing(labels) && return 0.0 # Cannot compute without labels
        equalized_odds(predictions, labels, protected)
    elseif value.metric == :equal_opportunity
        isnothing(labels) && return 0.0 # Cannot compute without labels
        equal_opportunity(predictions, labels, protected)
    elseif value.metric == :disparate_impact
        return disparate_impact(predictions, protected) # DI is a ratio, 1.0 is best
    elseif value.metric == :individual_fairness
        return 1.0 - individual_fairness(predictions, similarity_matrix) # Convert individual fairness to a maximization score
    else
        error("Unknown fairness metric: $(value.metric). Please ensure it is a valid metric from FairnessMetric enum.")
    end

    # For disparity metrics (demographic_parity, equalized_odds, equal_opportunity),
    # a lower disparity is better. Convert to a maximization score (1.0 - disparity).
    if value.metric in [:demographic_parity, :equalized_odds, :equal_opportunity]
        return 1.0 - disparity_or_score
    else
        return disparity_or_score # For disparate_impact and individual_fairness (already a score)
    end
end

"""
    maximize(value::Safety, initial_state::Dict)::Float64

Calculates a safety score for a given `Safety` objective based on the
`initial_state`. This function provides a scalar value that can be maximized
during an optimization process, representing the degree to which safety
conditions are met.

# Arguments
- `value::Safety`: A `Safety` object specifying the invariant and criticality.
- `initial_state::Dict`: A dictionary representing the current or candidate system state.
                         It should contain:
    - `:is_safe`: `Bool` - A general indication of the system's safety status.
                         Defaults to `true` if not provided.
    - `:invariant_holds`: `Bool` - A flag indicating whether the specific
                                  `value.invariant` is currently satisfied.
                                  Defaults to `true` if not provided.

# Returns
- `Float64`: The computed safety score. Returns `1.0` if the system is
             considered safe (both `:is_safe` and `:invariant_holds` are `true`
             in `initial_state`), and `0.0` otherwise.

# Example

```julia
safety_obj = Safety(invariant = "No system crash")
state_safe = Dict(:is_safe => true, :invariant_holds => true)
score_safe = maximize(safety_obj, state_safe) # Returns 1.0

state_unsafe = Dict(:is_safe => false, :invariant_holds => true)
score_unsafe = maximize(safety_obj, state_unsafe) # Returns 0.0
```
"""
function maximize(value::Safety, initial_state::Dict)::Float64
    is_safe = get(initial_state, :is_safe, true)
    invariant_holds = get(initial_state, :invariant_holds, true)
    return (is_safe && invariant_holds) ? 1.0 : 0.0
end

"""
    verify_value(value::Value, proof::Dict)::Bool

Verifies whether a formal proof object successfully demonstrates that an ML
system aligns with a specified `Value` objective. This function is a placeholder
for integrating formal verification tools and results into the Axiology framework.
It allows for an auditable and rigorous assessment of value compliance, moving
beyond empirical testing to mathematical guarantees.

# Arguments
- `value::Value`: The `Value` object (e.g., `Fairness`, `Safety`, `Welfare`)
                  for which value alignment is being verified.
- `proof::Dict`: A dictionary representing the formal proof object or verification
                 results. It must contain:
    - `:verified`: `Bool` - A flag indicating whether the formal proof was successful
                            in demonstrating value alignment. `true` if verified,
                            `false` otherwise.
                 It may optionally contain:
    - `:prover`: `Symbol` or `String` - The formal verification tool used (e.g., `:Lean`, `:Coq`, `:Isabelle`).
    - `:details`: `String` - Additional details or a summary of the proof's findings.

# Returns
- `Bool`: `true` if the formal proof successfully verifies value alignment (i.e.,
          `proof[:verified]` is `true`), `false` otherwise.

# Example

```julia
safety_invariant = Safety(invariant = "No harmful actions")
formal_proof_results = Dict(
    :verified => true,
    :prover => :Isabelle,
    :details => "Invariant 'No harmful actions' proven for all inputs in domain D."
)
println("Safety Invariant Verified: $(verify_value(safety_invariant, formal_proof_results))") # true
```
"""
function verify_value(value::Value, proof::Dict)::Bool
    verified = get(proof, :verified, false)
    return verified
end
