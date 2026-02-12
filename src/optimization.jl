# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Multi-Objective Optimization and Pareto Frontier Analysis for Value-Driven ML

This file provides tools for performing multi-objective optimization (MOO) and
Pareto frontier analysis within the Axiology.jl framework. In real-world machine
learning applications, models often need to satisfy multiple, often conflicting,
values simultaneously (e.g., maximizing profit while ensuring fairness and
maintaining efficiency). MOO is essential for understanding and managing these
inherent trade-offs.

Key Concepts:
-   **Competing Values**: Different `Value` objectives (Fairness, Welfare, Profit, etc.)
    rarely perfectly align. Improving one might degrade another.
-   **Value Scoring**: Functions like `value_score` normalize individual value
    fulfillment into a comparable `[0, 1]` range.
-   **Weighted Aggregation**: `weighted_score` allows combining multiple value
    scores into a single scalar objective using predefined weights.
-   **Pareto Dominance**: A solution is Pareto dominant over another if it is
    at least as good on all objectives and strictly better on at least one.
-   **Pareto Frontier**: The set of all non-dominated solutions. These solutions
    represent optimal trade-offs, where no objective can be improved without
    sacrificing another. Analyzing the Pareto frontier helps decision-makers
    choose the most appropriate solution based on their priorities.

The functions in this module facilitate:
1.  Quantifying how well a system state (e.g., an ML model's performance)
    satisfies various individual values.
2.  Aggregating these individual value scores into a composite score.
3.  Identifying the set of Pareto-optimal solutions from a collection of
    candidate system states, offering a clear view of the available trade-offs.
"""

"""
    value_score(value::Value, state::Dict)::Float64

Computes a normalized score for how well a specific `Value` objective is satisfied
in a given system `state`. The score is always a `Float64` between `0.0` and `1.0`,
where `1.0` represents optimal satisfaction of the value and `0.0` represents
complete dissatisfaction or inability to measure. This function is crucial
for comparing and aggregating diverse value objectives in multi-objective
optimization.

The method of scoring depends on the type of `value`:

# Arguments
- `value::Value`: The `Value` object (e.g., `Fairness`, `Welfare`, `Profit`, `Efficiency`, `Safety`)
                  for which to compute the score.
- `state::Dict`: A dictionary representing the current system state or evaluation results.
                 The required keys in this dictionary vary based on the `value` type.

# Returns
- `Float64`: A normalized score in the range `[0.0, 1.0]`, where `1.0` is optimal.

# Details for each Value type:
- `Fairness`:
    - **Required state keys**: `:predictions`, `:protected` (or `:protected_attributes`),
                                 optionally `:labels` (for `equalized_odds`, `equal_opportunity`),
                                 or `:similarity_matrix` (for `individual_fairness`).
    - **Score Calculation**: Typically `max(0.0, 1.0 - disparity / value.threshold)`.
                             A lower disparity (e.g., from `demographic_parity`)
                             results in a higher score. The score is clamped to `[0.0, 1.0]`.
                             For `disparate_impact`, it's `min(1.0, max(0.0, ratio / 0.8))`
                             (assuming 0.8 is the minimum acceptable ratio).
- `Welfare`:
    - **Required state keys**: `:utilities`.
    - **Score Calculation**: The computed welfare value (`utilitarian_welfare`,
                             `rawlsian_welfare`, or `egalitarian_welfare`) is normalized
                             by a `max_welfare` (from `state[:max_welfare]` or an
                             estimated maximum). The score is clamped to `[0.0, 1.0]`.
- `Profit`:
    - **Required state keys**: `:profit`.
    - **Score Calculation**: If `state[:profit]` meets `value.target`, returns `1.0`.
                             Otherwise, returns `state[:profit] / value.target`, clamped
                             to `[0.0, 1.0]`.
- `Efficiency`:
    - **Required state keys**: `:computation_time` (for `:computation_time` metric),
                                `:is_pareto_efficient` (for `:pareto` metric),
                                `:net_gain` (for `:kaldor_hicks` metric).
    - **Score Calculation**:
        - For `:computation_time`: `value.target / state[:computation_time]`,
                                    clamped to `[0.0, 1.0]` (lower time is better).
        - For `:pareto`: Returns `1.0` if `state[:is_pareto_efficient]` is `true`, `0.0` otherwise.
        - For `:kaldor_hicks`: Returns `state[:net_gain] / value.target`, clamped
                                to `[0.0, 1.0]`.
- `Safety`:
    - **Required state keys**: `:is_safe`, `:invariant_holds`.
    - **Score Calculation**: Returns `1.0` if `state[:is_safe]` is `true` AND
                             `state[:invariant_holds]` is `true`, `0.0` otherwise.

# Throws
- `ErrorException`: If required keys for a specific `value` type or metric are missing
                    from `state`.
- `ErrorException`: If an unknown `value` type or metric is encountered.

# Example

```julia
fairness = Fairness(metric = :demographic_parity, threshold = 0.05)
state_fair = Dict(:predictions => [1,0,1,0], :protected => [:a,:b,:a,:b])
score_fair = value_score(fairness, state_fair)

welfare = Welfare(metric = :utilitarian)
state_welfare = Dict(:utilities => [10.0, 8.0, 12.0, 7.0])
score_welfare = value_score(welfare, state_welfare) # Requires :max_welfare for proper normalization
```
"""
function value_score(value::Value, state::Dict)::Float64
    if value isa Fairness
        predictions = get(state, :predictions, nothing)
        protected = get(state, :protected, nothing)
        labels = get(state, :labels, nothing)
        similarity_matrix = get(state, :similarity_matrix, nothing)

        isnothing(predictions) && error("State must contain :predictions for Fairness value_score.")
        isnothing(protected) && value.metric != :individual_fairness && error("State must contain :protected or :protected_attributes for group Fairness value_score.")
        isnothing(similarity_matrix) && value.metric == :individual_fairness && error("State must contain :similarity_matrix for individual Fairness value_score.")


        disparity = if value.metric == :demographic_parity
            demographic_parity(predictions, protected)
        elseif value.metric == :equalized_odds
            isnothing(labels) && error("State must contain :labels for equalized_odds value_score.")
            equalized_odds(predictions, labels, protected)
        elseif value.metric == :equal_opportunity
            isnothing(labels) && error("State must contain :labels for equal_opportunity value_score.")
            equal_opportunity(predictions, labels, protected)
        elseif value.metric == :disparate_impact
            # For disparate impact, a ratio of 1.0 is optimal. Score is ratio / 1.0 (clamped).
            di_ratio = disparate_impact(predictions, protected)
            return min(1.0, max(0.0, di_ratio)) # Clamped to [0,1], 1.0 is best. Threshold (0.8) should be handled by satisfy
        elseif value.metric == :individual_fairness
            # For individual fairness, 0.0 is optimal (no difference for similar individuals).
            # Convert to a score where 1.0 is optimal. Assume max possible diff is 1.0.
            ind_fairness = individual_fairness(predictions, similarity_matrix)
            return max(0.0, 1.0 - ind_fairness)
        else
            error("Unknown fairness metric: $(value.metric) for value_score.")
        end

        # For disparity metrics, a lower disparity is better. Convert to score where 1.0 is optimal.
        # Normalize disparity relative to threshold. If disparity > threshold, score becomes < 0.
        return max(0.0, 1.0 - disparity / value.threshold)

    elseif value isa Welfare
        utilities = get(state, :utilities, nothing)
        isnothing(utilities) && error("State must contain :utilities for Welfare value_score.")

        # Handle empty utilities gracefully as in welfare functions
        if isempty(utilities)
             welfare_val = 0.0
        else
            welfare_val = if value.metric == :utilitarian
                utilitarian_welfare(utilities)
            elseif value.metric == :rawlsian
                rawlsian_welfare(utilities)
            elseif value.metric == :egalitarian
                egalitarian_welfare(utilities)
            else
                error("Unknown welfare metric: $(value.metric) for value_score.")
            end
        end

        # Normalize by an estimated maximum possible welfare.
        # This requires domain knowledge or an assumption.
        # Here, we assume maximum utility is `1.0` per individual, so `length(utilities)`
        # is a rough upper bound for utilitarian. For rawlsian, `1.0` is upper bound.
        # For egalitarian, `0.0` is the upper bound. This normalization is very context-dependent.
        max_possible_utility_per_individual = 1.0 # Assumption for normalization
        
        max_welfare_estimate = if value.metric == :utilitarian
            length(utilities) * max_possible_utility_per_individual
        elseif value.metric == :rawlsian
            max_possible_utility_per_individual
        elseif value.metric == :egalitarian
            0.0 # Max egalitarian score is 0.0 (zero variance)
        else
            1.0 # Fallback
        end

        if max_welfare_estimate == 0.0 && welfare_val == 0.0
            return 1.0 # Perfectly egalitarian, score is 1.0
        elseif max_welfare_estimate == 0.0
            return 0.0 # Error case or impossible to normalize
        end

        # Clamp to [0, 1] range after normalization
        # For egalitarian, max score is 0, so welfare_val is usually negative.
        # We want to map [min_possible_welfare, 0] to [0, 1].
        if value.metric == :egalitarian
            # If egalitarian_welfare returns 0.0, it's perfect equality (score 1.0).
            # If it's very negative, it's highly unequal (score 0.0).
            # This requires a lower bound for normalization, e.g., smallest possible -var.
            # For simplicity, we can use a heuristic or assume -100.0 as a floor for very unequal.
            # More robust would be to know the actual range of variance.
            # Here, we map negative welfare_val to 0 and 0 to 1.
            # For a variance of `v`, the score is `max(0, 1 + v / max_variance_expected)`.
            # If Var is 0, score 1. If Var is max_var_expected, score is 0.
            # Assuming max_var_expected could be 0.25 for utilities in [0,1]
            return max(0.0, 1.0 + welfare_val / 0.25) # Heuristic for now. Needs better scaling.
        else
            return min(1.0, max(0.0, welfare_val / max_welfare_estimate))
        end

    elseif value isa Profit
        profit = get(state, :profit, nothing)
        isnothing(profit) && error("State must contain :profit for Profit value_score.")

        if value.target == 0.0
            # If target is 0, any positive profit is optimal. Negative profit is worst.
            return profit >= 0.0 ? 1.0 : 0.0
        else
            return min(1.0, max(0.0, profit / value.target))
        end

    elseif value isa Efficiency
        if value.metric == :computation_time
            time = get(state, :computation_time, nothing)
            isnothing(time) && error("State must contain :computation_time for :computation_time efficiency value_score.")
            if time == 0.0
                return 1.0 # Instantly efficient
            else
                # Normalize by target. Lower time is better. Score 1.0 if time <= target.
                # If time is 2*target, score is 0.5.
                return min(1.0, max(0.0, value.target / time))
            end
        elseif value.metric == :pareto
            is_pareto = get(state, :is_pareto_efficient, nothing)
            isnothing(is_pareto) && error("State must contain :is_pareto_efficient for :pareto efficiency value_score.")
            return is_pareto ? 1.0 : 0.0
        elseif value.metric == :kaldor_hicks
            net_gain = get(state, :net_gain, nothing)
            isnothing(net_gain) && error("State must contain :net_gain for :kaldor_hicks efficiency value_score.")
            # Normalize net_gain by target
            if value.target == 0.0
                return net_gain >= 0.0 ? 1.0 : 0.0
            else
                return min(1.0, max(0.0, net_gain / value.target))
            end
        else
            error("Unknown efficiency metric: $(value.metric) for value_score.")
        end

    elseif value isa Safety
        is_safe = get(state, :is_safe, true)
        invariant_holds = get(state, :invariant_holds, true)
        return (is_safe && invariant_holds) ? 1.0 : 0.0

    else
        error("Unknown Value type: $(typeof(value)) for value_score.")
    end
end

"""
    weighted_score(values::Vector{<:Value}, state::Dict)::Float64

Computes a single, aggregated weighted score across multiple `Value` objectives
for a given system `state`. This function is fundamental for multi-objective
optimization, allowing the prioritization of different values based on their
assigned weights.

The weighted score `S` is calculated as:
`S = (Σᵢ score(valueᵢ, state) * weightᵢ) / (Σᵢ weightᵢ)`

# Arguments
- `values::Vector{<:Value}`: A vector of `Value` objects, each with its
                              own `weight` attribute, representing the
                              objectives to be aggregated.
- `state::Dict`: A dictionary representing the current system state. This
                 dictionary must contain all necessary information for
                 `value_score` to compute a score for each `Value` in `values`.

# Returns
- `Float64`: A composite score in the range `[0.0, 1.0]`, representing the
             overall satisfaction of the weighted values in the given `state`.

# Details:
- Each `value` in the `values` vector is first converted into a normalized
  score using `value_score(v, state)`.
- These normalized scores are then multiplied by their respective `v.weight`.
- The sum of these weighted scores is then divided by the `total_weight` (sum
  of all `v.weight`) to produce a final normalized composite score.
- If `total_weight` is `0.0` (meaning all values have a weight of `0.0`), the
  function returns `0.0`.

# Throws
- `ErrorException`: If `value_score` encounters an error due to missing keys
                    in `state` for a particular `Value`.

# Example

```julia
values_to_aggregate = [
    Fairness(metric = :demographic_parity, threshold = 0.05, weight = 0.4),
    Welfare(metric = :utilitarian, weight = 0.3),
    Efficiency(metric = :computation_time, target = 0.1, weight = 0.3)
]

# Ensure state contains data required by each value_score calculation
state_example = Dict(
    :predictions => [1, 0, 1, 1, 0, 1],
    :protected => [:male, :female, :male, :female, :male, :female],
    :utilities => [10.0, 8.0, 12.0, 7.0, 9.0, 11.0],
    :computation_time => 0.08, # Actual time
    :max_welfare => 60.0 # Example max welfare for normalization
)

score = weighted_score(values_to_aggregate, state_example)
println("Aggregated Weighted Score: $(score)")
```
"""
function weighted_score(values::Vector{<:Value}, state::Dict)::Float64
    total_weight = sum(v.weight for v in values)

    if total_weight == 0.0
        # If all weights are zero, the aggregated score is 0.0 as no value contributes.
        return 0.0
    end

    weighted_sum = sum(value_score(v, state) * v.weight for v in values)
    return weighted_sum / total_weight
end

"""
    normalize_scores(scores::AbstractVector{<:Real})::Vector{Float64}

Normalizes a vector of raw scores to the `[0, 1]` range using min-max scaling.
This transformation is often necessary in multi-objective optimization to bring
different objectives to a comparable scale before aggregation or Pareto analysis.

The formula for min-max scaling is:
`normalized_score = (score - min(scores)) / (max(scores) - min(scores))`

# Arguments
- `scores::AbstractVector{<:Real}`: A vector of raw numerical scores to be normalized.

# Returns
- `Vector{Float64}`: A new vector containing the normalized scores, each in the
                     range `[0.0, 1.0]`.

# Details:
- If all scores in the input vector are identical (`max_score == min_score`),
  the function returns a vector of `1.0`s, indicating that all elements are
  at the maximum relative to each other.
- If the `scores` vector is empty, this function will throw an error as `minimum`
  and `maximum` are not defined for empty collections.

# Throws
- `ArgumentError`: If the input `scores` vector is empty.

# Example

```julia
raw_scores = [10.0, 20.0, 5.0, 15.0]
normalized = normalize_scores(raw_scores) # Returns [0.5, 1.0, 0.0, 0.75]

identical_scores = [5.0, 5.0, 5.0]
normalized_identical = normalize_scores(identical_scores) # Returns [1.0, 1.0, 1.0]
```
"""
function normalize_scores(scores::AbstractVector{<:Real})::Vector{Float64}
    if isempty(scores)
        throw(ArgumentError("Cannot normalize an empty vector of scores."))
    end

    min_score = minimum(scores)
    max_score = maximum(scores)

    if max_score == min_score
        return ones(length(scores))
    end

    return [(s - min_score) / (max_score - min_score) for s in scores]
end

"""
    dominated(solution_a::Dict, solution_b::Dict, values::AbstractVector{<:Value})::Bool

Determines if `solution_a` is Pareto dominated by `solution_b` with respect
to a given set of `values`. A solution `A` is said to be Pareto dominated by
another solution `B` if `B` is at least as good as `A` on all objectives
(values) and strictly better than `A` on at least one objective. This concept
is fundamental to identifying the Pareto frontier in multi-objective optimization.

# Arguments
- `solution_a::Dict`: A dictionary representing the first solution's state.
                      This dictionary must contain all necessary information
                      for `value_score` to compute scores for each `Value` in `values`.
- `solution_b::Dict`: A dictionary representing the second solution's state.
                      This dictionary must contain all necessary information
                      for `value_score` to compute scores for each `Value` in `values`.
- `values::AbstractVector{<:Value}`: A vector of `Value` objects, representing
                                      the objectives (criteria) used for comparison.

# Returns
- `Bool`: `true` if `solution_a` is Pareto dominated by `solution_b`, `false` otherwise.

# Logic:
The function iterates through each `value` (objective) and compares the
`value_score` for both solutions:
- `better_on_all` is `true` initially and becomes `false` if `solution_b` is
   worse than `solution_a` on any `value`.
- `strictly_better_on_one` becomes `true` if `solution_b` is strictly better
   than `solution_a` on at least one `value`.
`solution_a` is dominated by `solution_b` if and only if `better_on_all` is
`true` AND `strictly_better_on_one` is `true` after checking all values.

# Throws
- `ErrorException`: If `value_score` fails for either solution due to missing
                    required keys in their respective `Dict`s.

# Example

```julia
fairness = Fairness(metric = :demographic_parity, threshold = 0.05)
welfare = Welfare(metric = :utilitarian)
objectives = [fairness, welfare]

# Assume these dicts contain data for value_score to operate
sol1 = Dict(:predictions => [0.1], :protected => [:a], :utilities => [10.0], :max_welfare => 100.0)
sol2 = Dict(:predictions => [0.05], :protected => [:a], :utilities => [15.0], :max_welfare => 100.0)

# If sol2 has better or equal fairness AND better or equal welfare, and strictly better on one
is_dominated = dominated(sol1, sol2, objectives)
println("Solution 1 dominated by Solution 2: $(is_dominated)")
```
"""
function dominated(solution_a::Dict, solution_b::Dict, values::AbstractVector{<:Value})::Bool
    better_on_all = true
    strictly_better_on_one = false

    for value in values
        score_a = value_score(value, solution_a)
        score_b = value_score(value, solution_b)

        if score_b < score_a # solution_b is worse on this value
            better_on_all = false
            break # No need to check further, A is not dominated by B
        elseif score_b > score_a # solution_b is strictly better on this value
            strictly_better_on_one = true
        end
    end

    return better_on_all && strictly_better_on_one
end

"""
    pareto_frontier(solutions::Vector{<:Dict}, values::AbstractVector{<:Value})::Vector{Dict}

Computes the Pareto frontier from a given set of `solutions`. The Pareto frontier
consists of all non-dominated solutions, meaning solutions for which no other
solution in the set is strictly better on at least one objective (value) and
at least as good on all other objectives. These solutions represent optimal
trade-offs among the competing values.

Identifying the Pareto frontier is a crucial step in multi-objective optimization,
as it presents decision-makers with the set of best possible compromises.

# Arguments
- `solutions::Vector{<:Dict}`: A vector where each `Dict` represents a candidate
                               system state or configuration. Each `Dict` must
                               contain all necessary information for `value_score`
                               to compute scores for each `Value` in `values`.
- `values::AbstractVector{<:Value}`: A vector of `Value` objects, representing
                                      the objectives (criteria) against which
                                      solutions are compared.

# Returns
- `Vector{Dict}`: A new vector containing only the Pareto-optimal solutions
                  from the input `solutions` set.

# Algorithm:
1.  Initialize an empty list `pareto_optimal`.
2.  For each `solution` in the input `solutions`:
    a.  Assume `solution` is not dominated (`is_dominated = false`).
    b.  Compare `solution` against every `other` solution in the set.
    c.  If `solution` is distinct from `other` and `dominated(solution, other, values)`
        returns `true` (meaning `solution` is dominated by `other`), then set
        `is_dominated = true` and break from the inner loop.
    d.  If `is_dominated` remains `false` after comparison with all other solutions,
        add `solution` to the `pareto_optimal` list.
3.  Return `pareto_optimal`.

# Throws
- `ErrorException`: If `value_score` (called internally by `dominated`) fails
                    for any solution due to missing required keys.

# Example

```julia
solutions_candidates = [
    # Example structure of solution Dicts:
    Dict(:id => 1, :predictions => [0.1], :protected => [:a], :utilities => [10.0], :computation_time => 0.1, :max_welfare => 100.0),
    Dict(:id => 2, :predictions => [0.05], :protected => [:a], :utilities => [12.0], :computation_time => 0.15, :max_welfare => 100.0),
    Dict(:id => 3, :predictions => [0.15], :protected => [:a], :utilities => [8.0], :computation_time => 0.08, :max_welfare => 100.0)
]

fairness = Fairness(metric = :demographic_parity, threshold = 0.05)
welfare = Welfare(metric = :utilitarian)
efficiency = Efficiency(metric = :computation_time, target = 0.1)

values_to_optimize = [fairness, welfare, efficiency]

pareto_solutions = pareto_frontier(solutions_candidates, values_to_optimize)
println("Pareto-Optimal Solutions: $(pareto_solutions)")
```
"""
function pareto_frontier(solutions::Vector{<:Dict}, values::AbstractVector{<:Value})::Vector{Dict}
    if isempty(solutions)
        return eltype(solutions)[]
    end

    pareto_optimal = eltype(solutions)[]

    for solution in solutions
        is_dominated = false

        for other in solutions
            # Ensure solution !== other to avoid self-comparison
            if solution !== other && dominated(solution, other, values)
                is_dominated = true
                break
            end
        end

        if !is_dominated
            push!(pareto_optimal, solution)
        end
    end

    return pareto_optimal
end

"""
    pareto_frontier(system::Dict, values::AbstractVector{<:Value})::Vector{Dict}

A convenience wrapper for `pareto_frontier(solutions::Vector{Dict}, ...)` that
allows computing the Pareto frontier directly from a `system` specification.
This method is intended for scenarios where the "system" itself can either
provide a predefined set of candidate solutions or implicitly represents
a single candidate solution.

# Arguments
- `system::Dict`: A dictionary representing the system under consideration.
                  It can be structured in two ways:
    1.  As a container for candidate solutions: If `system` has a `:solutions` key,
        its value (expected to be `Vector{Dict}`) will be used as the set of
        candidate solutions. This is useful when the system itself generates or
        manages a collection of possible configurations/states.
    2.  As a single candidate solution: If `system` does not have a `:solutions` key,
        the `system` `Dict` itself will be treated as the sole candidate solution.
        This allows evaluating a single system state's position relative to
        a hypothetical Pareto frontier (which, for a single point, is trivially itself).
- `values::AbstractVector{<:Value}`: A vector of `Value` objects, representing
                                      the objectives against which solutions are compared.

# Returns
- `Vector{Dict}`: A vector containing the Pareto-optimal solutions identified
                  from the provided `system` (either its `:solutions` or the `system` itself).

# Use Cases:
- **Parameter Sweeps**: When `system[:solutions]` contains results from a parameter
                        sweep or grid search, this function can find the non-dominated
                        configurations.
- **Single System Evaluation**: Quickly determine if a single ML model configuration
                                is non-dominated when compared only against itself.

# Throws
- `ErrorException`: If `value_score` (called internally) fails for any solution
                    due to missing required keys in their respective `Dict`s.

# Example

```julia
# Example 1: System with predefined candidate solutions
system_with_candidates = Dict(
    :solutions => [
        Dict(:id => "ConfigA", :utilities => [10.0], :computation_time => 0.1),
        Dict(:id => "ConfigB", :utilities => [12.0], :computation_time => 0.15)
    ]
)
values_obj = [Welfare(metric = :utilitarian), Efficiency(metric = :computation_time, target = 0.1)]
pareto_sol = pareto_frontier(system_with_candidates, values_obj)

# Example 2: System as a single candidate solution
single_system_state = Dict(:id => "MyModel", :utilities => [15.0], :computation_time => 0.05)
pareto_single = pareto_frontier(single_system_state, values_obj)
```
"""
function pareto_frontier(system::Dict, values::AbstractVector{<:Value})::Vector{Dict}
    # Generate candidate solutions by exploring the parameter space
    solutions = Dict[]

    # If system provides candidate solutions, use them
    if haskey(system, :solutions)
        solutions = system[:solutions]
    else
        # Otherwise, just evaluate the current system state
        push!(solutions, system)
    end

    # Call the primary pareto_frontier method
    return pareto_frontier(solutions, values)
end
