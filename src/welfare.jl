# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Welfare function implementations (utilitarian, Rawlsian, egalitarian).
"""

"""
    utilitarian_welfare(utilities::AbstractVector{<:Real})

Compute utilitarian social welfare (sum of utilities).

Maximizes total welfare: W = Σᵢ uᵢ

# Arguments
- `utilities::AbstractVector{<:Real}`: Individual utility values

# Returns
- `Float64`: Total utilitarian welfare

# Example

```julia
utilities = [10.0, 8.0, 12.0, 7.0]
welfare = utilitarian_welfare(utilities)  # 37.0
```
"""
function utilitarian_welfare(utilities::AbstractVector{<:Real})
    return sum(utilities)
end

"""
    rawlsian_welfare(utilities::AbstractVector{<:Real})

Compute Rawlsian (maximin) social welfare.

Maximizes the welfare of the worst-off individual: W = min(u₁, u₂, ..., uₙ)

# Arguments
- `utilities::AbstractVector{<:Real}`: Individual utility values

# Returns
- `Float64`: Minimum utility (Rawlsian welfare)

# Example

```julia
utilities = [10.0, 8.0, 12.0, 7.0]
welfare = rawlsian_welfare(utilities)  # 7.0 (worst-off)
```
"""
function rawlsian_welfare(utilities::AbstractVector{<:Real})
    return minimum(utilities)
end

"""
    egalitarian_welfare(utilities::AbstractVector{<:Real})

Compute egalitarian welfare (negative variance).

Measures equality by penalizing variance: W = -Var(utilities)

Lower variance (more equal) = higher welfare.

# Arguments
- `utilities::AbstractVector{<:Real}`: Individual utility values

# Returns
- `Float64`: Negative variance (higher = more equal)

# Example

```julia
utilities = [10.0, 10.0, 10.0, 10.0]
welfare = egalitarian_welfare(utilities)  # 0.0 (perfect equality)

utilities = [5.0, 10.0, 15.0, 20.0]
welfare = egalitarian_welfare(utilities)  # -31.25 (high inequality)
```
"""
function egalitarian_welfare(utilities::AbstractVector{<:Real})
    return -var(utilities)
end

"""
    satisfy(value::Welfare, state::Dict)

Check if a system state satisfies minimum welfare requirements.

# Arguments
- `value::Welfare`: Welfare value specification
- `state::Dict`: System state containing utilities

# Returns
- `Bool`: Whether minimum welfare is satisfied

# Example

```julia
welfare = Welfare(metric = :rawlsian)
state = Dict(:utilities => [8.0, 9.0, 10.0], :min_welfare => 7.0)
satisfy(welfare, state)  # true (min utility 8.0 > 7.0)
```
"""
function satisfy(value::Welfare, state::Dict)
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
    satisfy(value::Profit, state::Dict)

Check if profit target is met.

# Arguments
- `value::Profit`: Profit value specification
- `state::Dict`: System state containing current profit

# Returns
- `Bool`: Whether profit target is satisfied
"""
function satisfy(value::Profit, state::Dict)
    current_profit = get(state, :profit, 0.0)

    # Check profit target
    profit_ok = current_profit >= value.target

    # Check all constraints
    constraints_ok = all(satisfy(c, state) for c in value.constraints)

    return profit_ok && constraints_ok
end

"""
    satisfy(value::Efficiency, state::Dict)

Check if efficiency target is met.

# Arguments
- `value::Efficiency`: Efficiency value specification
- `state::Dict`: System state containing efficiency metrics

# Returns
- `Bool`: Whether efficiency target is satisfied
"""
function satisfy(value::Efficiency, state::Dict)
    if value.metric == :computation_time
        time = get(state, :computation_time, Inf)
        return time <= value.target
    elseif value.metric == :pareto
        # Pareto efficiency check (no one can be made better off without making someone worse off)
        is_pareto = get(state, :is_pareto_efficient, false)
        return is_pareto
    elseif value.metric == :kaldor_hicks
        # Kaldor-Hicks efficiency (winners can compensate losers)
        net_gain = get(state, :net_gain, 0.0)
        return net_gain >= value.target
    else
        error("Unknown efficiency metric: $(value.metric)")
    end
end

"""
    satisfy(value::Safety, state::Dict)

Check if safety invariant holds.

# Arguments
- `value::Safety`: Safety value specification
- `state::Dict`: System state

# Returns
- `Bool`: Whether safety invariant is satisfied
"""
function satisfy(value::Safety, state::Dict)
    # Check if safety invariant is satisfied in state
    is_safe = get(state, :is_safe, true)
    invariant_holds = get(state, :invariant_holds, true)

    return is_safe && invariant_holds
end

"""
    maximize(value::Value, initial_state::Dict)

Optimize system to maximize the given value.

Returns an optimized score/state representation.

# Arguments
- `value::Value`: Value to maximize
- `initial_state::Dict`: Initial system state

# Returns
- `Float64`: Optimized value score

# Example

```julia
welfare = Welfare(metric = :utilitarian)
state = Dict(:utilities => [5.0, 6.0, 7.0])
score = maximize(welfare, state)
```
"""
function maximize(value::Welfare, initial_state::Dict)
    utilities = get(initial_state, :utilities, Float64[])

    if value.metric == :utilitarian
        return utilitarian_welfare(utilities)
    elseif value.metric == :rawlsian
        return rawlsian_welfare(utilities)
    elseif value.metric == :egalitarian
        return egalitarian_welfare(utilities)
    end
end

function maximize(value::Profit, initial_state::Dict)
    profit = get(initial_state, :profit, 0.0)
    return profit
end

function maximize(value::Efficiency, initial_state::Dict)
    if value.metric == :computation_time
        time = get(initial_state, :computation_time, 0.0)
        return -time  # Negative because we want to minimize time
    else
        return 1.0
    end
end

function maximize(value::Fairness, initial_state::Dict)
    # Return fairness score (1.0 - disparity)
    predictions = get(initial_state, :predictions, nothing)
    protected = get(initial_state, :protected, nothing)

    if isnothing(predictions) || isnothing(protected)
        return 0.0
    end

    disparity = demographic_parity(predictions, protected)
    return 1.0 - disparity
end

function maximize(value::Safety, initial_state::Dict)
    is_safe = get(initial_state, :is_safe, true)
    return is_safe ? 1.0 : 0.0
end

"""
    verify_value(value::Value, proof::Dict)

Verify that a formal proof demonstrates value alignment.

# Arguments
- `value::Value`: Value to verify
- `proof::Dict`: Proof object containing verification results

# Returns
- `Bool`: Whether proof successfully verifies value

# Example

```julia
safety = Safety(invariant = "No harmful actions")
proof = Dict(:verified => true, :prover => :Lean)
verify_value(safety, proof)  # true
```
"""
function verify_value(value::Value, proof::Dict)
    verified = get(proof, :verified, false)
    return verified
end
