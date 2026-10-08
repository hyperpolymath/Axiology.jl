# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

"""
    _require_nonzero_target(value::Value, field::AbstractString)

Throw an `ArgumentError` explaining that a score normalised by a zero `field`
is undefined, instead of letting the division return `Inf` or `NaN`.
"""
function _require_nonzero_target(value::Value, field::AbstractString)
    throw(ArgumentError("value_score($(nameof(typeof(value)))) divides by `$(field)`, " *
                        "which is 0.0, so the score is undefined; construct the value " *
                        "with a non-zero `$(field)` to score it."))
end

"""
    value_score(value::Value, state::Dict)::Float64

Score how well `state` does on `value`, higher is better. The scale depends on
the value type:

- `Fairness`: `1 - disparity / threshold`, floored at `0.0`, so `1.0` is perfect
  parity and `0.0` means the disparity reached the threshold. With
  `threshold == 0.0` only exact parity scores `1.0`; anything else scores `0.0`.
  `:disparate_impact` scores the ratio itself (clamped to `[0, 1]`) and
  `:individual_fairness` scores `1 - mean difference`.
- `Welfare`: the welfare function of `:utilities`, scaled to `[0, 1]` by
  `:max_welfare` when the state provides a positive one; otherwise the **raw,
  unbounded** welfare value.
- `Profit`: `:profit / target`.
- `Efficiency`: `1 - :computation_time / target` (floored at `0.0`),
  `1.0`/`0.0` for `:is_pareto_efficient`, or `:net_gain / target`.
- `Safety`: `1.0` when safe, else `0.0` (absent keys count as safe, as in
  `satisfy(::Safety, ::Dict)`).

The group fairness metrics read `:protected`, or `:protected_attributes` when
`:protected` is absent.

Scores normalised by a target (`Profit`, `Efficiency` `:computation_time` and
`:kaldor_hicks`) throw an `ArgumentError` when that target is `0.0` rather than
returning `Inf` or `NaN`. Note that `Profit()` has `target = 0.0` by default.
"""
function value_score(value::Value, state::Dict)::Float64
    if value isa Fairness
        predictions = get(state, :predictions, nothing)
        protected = get(state, :protected, get(state, :protected_attributes, nothing))
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
        # A zero threshold tolerates no disparity at all: 0/0 would otherwise give NaN.
        iszero(value.threshold) && return iszero(disparity) ? 1.0 : 0.0
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

        # Normalize welfare value to [0,1] using max_welfare from state if provided.
        # Without max_welfare, return the raw welfare value so callers (e.g. pareto_frontier)
        # can compare solutions meaningfully rather than collapsing all positive values to 1.0.
        max_welfare = get(state, :max_welfare, nothing)
        if !isnothing(max_welfare) && max_welfare > 0.0
            return min(1.0, max(0.0, welfare_val / max_welfare))
        else
            return welfare_val
        end

    elseif value isa Profit
        profit = get(state, :profit, nothing)
        isnothing(profit) && error("State must contain :profit for Profit value_score.")
        iszero(value.target) && _require_nonzero_target(value, "target")
        # Normalize profit relative to target
        return profit / value.target

    elseif value isa Efficiency
        if value.metric == :computation_time
            time = get(state, :computation_time, nothing)
            isnothing(time) && error("State must contain :computation_time for Efficiency value_score.")
            iszero(value.target) && _require_nonzero_target(value, "target")
            # Lower time is better - invert and normalize
            return max(0.0, 1.0 - time / value.target)
        elseif value.metric == :pareto
            is_pareto = get(state, :is_pareto_efficient, nothing)
            isnothing(is_pareto) && error("State must contain :is_pareto_efficient for Efficiency value_score.")
            return is_pareto ? 1.0 : 0.0
        elseif value.metric == :kaldor_hicks
            net_gain = get(state, :net_gain, nothing)
            isnothing(net_gain) && error("State must contain :net_gain for Efficiency value_score.")
            iszero(value.target) && _require_nonzero_target(value, "target")
            return net_gain / value.target
        else
            error("Unknown efficiency metric: $(value.metric) for value_score.")
        end

    elseif value isa Safety
        is_safe = get(state, :is_safe, true)
        invariant_holds = get(state, :invariant_holds, true)
        return (is_safe && invariant_holds) ? 1.0 : 0.0

    else
        error("Unknown value type: $(typeof(value)) for value_score.")
    end
end

"""
    weighted_score(values::Vector{<:Value}, state::Dict)::Float64

Return the weight-averaged `value_score` of `state` across `values`, using each
value's `weight`. Returns `0.0` when every weight is zero.

This collapses several objectives into one number, so it is only meaningful
when the individual scores share a scale; un-normalised `Welfare` scores (no
`:max_welfare` in `state`) and `Profit`/`Efficiency` ratios can dominate the
average. Use `dominated`/`pareto_frontier` to compare without collapsing.
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
    normalize_scores(scores::AbstractVector)::Vector{Float64}

Min-max normalise `scores` to `[0, 1]`. If every score is equal the result is
all ones. Throws an `ArgumentError` for an empty vector.
"""
function normalize_scores(scores::AbstractVector)::Vector{Float64}
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

Return whether `solution_a` is Pareto-dominated by `solution_b`: `solution_b`
scores at least as well as `solution_a` on every value in `values` and strictly
better on at least one, comparing `value_score`s.
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
    pareto_frontier(system::Dict, values::AbstractVector{<:Value})::Vector{Dict}

Return the solutions that no other solution dominates on `values` (see
`dominated`), in their original order.

The `system::Dict` form takes candidate solutions from `system[:solutions]`, or
treats `system` itself as the only candidate. Neither form generates
candidates; it filters the ones supplied. Comparison is pairwise, `O(n²)` in
the number of solutions.
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

function pareto_frontier(system::Dict, values::AbstractVector{<:Value})::Vector{Dict}
    # Candidate solutions are supplied by the caller; none are generated here.
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
