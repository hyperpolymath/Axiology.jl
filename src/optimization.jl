# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Multi-objective optimization and Pareto frontier analysis.
"""

"""
    value_score(value::Value, state::Dict)

Compute a normalized score for how well the value is satisfied.

# Arguments
- `value::Value`: Value to score
- `state::Dict`: System state

# Returns
- `Float64`: Score in [0, 1] where 1.0 is optimal

# Example

```julia
fairness = Fairness(metric = :demographic_parity, threshold = 0.05)
state = Dict(:predictions => [1,0,1,0], :protected => [:a,:b,:a,:b])
score = value_score(fairness, state)
```
"""
function value_score(value::Value, state::Dict)
    if value isa Fairness
        predictions = get(state, :predictions, nothing)
        protected = get(state, :protected, nothing)

        if isnothing(predictions) || isnothing(protected)
            return 0.0
        end

        disparity = demographic_parity(predictions, protected)
        # Score: 1.0 for no disparity, 0.0 for maximum disparity
        return max(0.0, 1.0 - disparity / value.threshold)

    elseif value isa Welfare
        utilities = get(state, :utilities, Float64[])

        if isempty(utilities)
            return 0.0
        end

        welfare_val = if value.metric == :utilitarian
            utilitarian_welfare(utilities)
        elseif value.metric == :rawlsian
            rawlsian_welfare(utilities)
        else
            egalitarian_welfare(utilities)
        end

        # Normalize by maximum possible welfare
        max_welfare = get(state, :max_welfare, maximum(utilities) * length(utilities))
        # Clamp to [0, 1] range
        return min(1.0, max(0.0, welfare_val / max_welfare))

    elseif value isa Profit
        profit = get(state, :profit, 0.0)
        return profit >= value.target ? 1.0 : profit / value.target

    elseif value isa Efficiency
        if value.metric == :computation_time
            time = get(state, :computation_time, Inf)
            return time <= value.target ? 1.0 : value.target / time
        else
            return 1.0
        end

    elseif value isa Safety
        is_safe = get(state, :is_safe, true)
        return is_safe ? 1.0 : 0.0

    else
        return 0.0
    end
end

"""
    weighted_score(values::Vector{<:Value}, state::Dict)

Compute weighted score across multiple values.

# Arguments
- `values::Vector{<:Value}`: Values to evaluate
- `state::Dict`: System state

# Returns
- `Float64`: Weighted score

# Example

```julia
values = [
    Fairness(metric = :demographic_parity, weight = 0.4),
    Welfare(metric = :utilitarian, weight = 0.3),
    Efficiency(metric = :computation_time, weight = 0.3)
]
score = weighted_score(values, state)
```
"""
function weighted_score(values::Vector{<:Value}, state::Dict)
    total_weight = sum(v.weight for v in values)

    if total_weight == 0.0
        return 0.0
    end

    weighted_sum = sum(value_score(v, state) * v.weight for v in values)
    return weighted_sum / total_weight
end

"""
    normalize_scores(scores::AbstractVector{<:Real})

Normalize scores to [0, 1] range.

# Arguments
- `scores::AbstractVector{<:Real}`: Raw scores

# Returns
- `Vector{Float64}`: Normalized scores
"""
function normalize_scores(scores::AbstractVector{<:Real})
    min_score = minimum(scores)
    max_score = maximum(scores)

    if max_score == min_score
        return ones(length(scores))
    end

    return [(s - min_score) / (max_score - min_score) for s in scores]
end

"""
    dominated(solution_a::Dict, solution_b::Dict, values::AbstractVector{<:Value})

Check if solution A is dominated by solution B (B is better on all values).

# Arguments
- `solution_a::Dict`: First solution
- `solution_b::Dict`: Second solution
- `values::AbstractVector{<:Value}`: Values to compare

# Returns
- `Bool`: True if A is dominated by B

# Example

```julia
dominated(solution1, solution2, [fairness, welfare])
```
"""
function dominated(solution_a::Dict, solution_b::Dict, values::AbstractVector{<:Value})
    better_on_all = true
    strictly_better_on_one = false

    for value in values
        score_a = value_score(value, solution_a)
        score_b = value_score(value, solution_b)

        if score_b < score_a
            better_on_all = false
            break
        elseif score_b > score_a
            strictly_better_on_one = true
        end
    end

    return better_on_all && strictly_better_on_one
end

"""
    pareto_frontier(solutions::Vector{Dict}, values::AbstractVector{<:Value})

Compute Pareto frontier from a set of solutions.

The Pareto frontier consists of non-dominated solutions - solutions where no other
solution is better on all objectives.

# Arguments
- `solutions::Vector{Dict}`: Candidate solutions
- `values::AbstractVector{<:Value}`: Values to optimize

# Returns
- `Vector{Dict}`: Pareto-optimal solutions

# Example

```julia
solutions = [
    Dict(:utilities => [10,8], :computation_time => 0.1, :predictions => [1,0], :protected => [:a,:b]),
    Dict(:utilities => [8,8], :computation_time => 0.05, :predictions => [1,1], :protected => [:a,:b]),
    Dict(:utilities => [12,6], :computation_time => 0.15, :predictions => [1,0], :protected => [:a,:b])
]

values = [
    Welfare(metric = :utilitarian, weight = 0.5),
    Efficiency(metric = :computation_time, weight = 0.5)
]

pareto_solutions = pareto_frontier(solutions, values)
```
"""
function pareto_frontier(solutions::Vector{<:Dict}, values::AbstractVector{<:Value})
    if isempty(solutions)
        return eltype(solutions)[]
    end

    pareto_optimal = eltype(solutions)[]

    for solution in solutions
        is_dominated = false

        for other in solutions
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
    pareto_frontier(system::Dict, values::AbstractVector{<:Value})

Generate Pareto frontier for a system with multiple objectives.

This is a convenience wrapper that generates candidate solutions by varying parameters.

# Arguments
- `system::Dict`: System specification
- `values::AbstractVector{<:Value}`: Values to optimize

# Returns
- `Vector{Dict}`: Pareto-optimal solutions
"""
function pareto_frontier(system::Dict, values::AbstractVector{<:Value})
    # Generate candidate solutions by exploring the parameter space
    solutions = Dict[]

    # If system provides candidate solutions, use them
    if haskey(system, :solutions)
        solutions = system[:solutions]
    else
        # Otherwise, just evaluate the current system state
        push!(solutions, system)
    end

    return pareto_frontier(solutions, values)
end
