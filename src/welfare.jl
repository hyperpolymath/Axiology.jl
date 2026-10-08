# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

using Statistics

"""
    utilitarian_welfare(utilities::AbstractVector)::Float64

Return the utilitarian (Benthamite) social welfare: the sum of individual
utilities. An empty vector has welfare `0.0`.
"""
function utilitarian_welfare(utilities::AbstractVector)::Float64
    isempty(utilities) && return 0.0 # Return 0.0 for empty utility vector
    return sum(utilities)
end

"""
    rawlsian_welfare(utilities::AbstractVector)::Float64

Return the Rawlsian (maximin) social welfare: the utility of the worst-off
individual. Raises an error for an empty vector, where no worst-off individual
exists.
"""
function rawlsian_welfare(utilities::AbstractVector)::Float64
    isempty(utilities) && error("Cannot compute Rawlsian welfare for an empty utility vector.")
    return minimum(utilities)
end

"""
    egalitarian_welfare(utilities::AbstractVector{<:Real})::Float64

Return an egalitarian welfare score: the negated sample variance of the
utilities, so that more equal distributions score higher and perfect equality
scores `0.0`. Fewer than two individuals score `0.0`.

This measures equality only; it does not reward higher utility levels.
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

"""
    satisfy(value::Value, state::Dict)::Bool

Return whether the system state `state` meets the criterion expressed by
`value`. Each `Value` subtype reads its own keys from `state`:

- `Fairness`: see `satisfy(::Fairness, ::Dict)`.
- `Welfare`: `:utilities` (required); passes when the chosen welfare function is
  at least `:min_welfare` (default `0.0`).
- `Profit`: `:profit` (required); passes when it reaches `value.target` and every
  `value.constraints` entry is itself satisfied by `state`.
- `Efficiency`: `:computation_time` (`<= target`), `:is_pareto_efficient`
  (must be `true`) or `:net_gain` (`>= target`), according to `value.metric`.
- `Safety`: `:is_safe` and `:invariant_holds`. **Both default to `true` when
  absent**, so an empty state counts as safe; supply them explicitly. The
  `value.invariant` text is descriptive and is not evaluated.

Missing required keys raise an error.
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

"""
    satisfy(value::Profit, state::Dict)::Bool

Return whether `state[:profit]` reaches `value.target` and every constraint in
`value.constraints` is satisfied by the same `state`.
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

Return whether `state` meets the efficiency criterion: `:computation_time <=
target`, `:is_pareto_efficient == true`, or `:net_gain >= target`, according to
`value.metric`.
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

Return `state[:is_safe] && state[:invariant_holds]`. Both keys default to
`true` when absent, so an empty state is reported as safe; callers must supply
them. `value.invariant` is not evaluated.
"""
function satisfy(value::Safety, state::Dict)::Bool
    # Check if safety invariant is satisfied in state
    is_safe = get(state, :is_safe, true) # Optimistic assumption if not provided
    invariant_holds = get(state, :invariant_holds, true) # Optimistic assumption if not provided

    return is_safe && invariant_holds
end

"""
    maximize(value::Value, initial_state::Dict)::Float64

Return the objective that `value` would maximise, evaluated at `initial_state`
(higher is better). This *evaluates* the objective at the given state; it does
not search a configuration space or change the state.

- `Welfare`: the chosen welfare function of `:utilities`.
- `Profit`: `:profit`.
- `Efficiency`: `-:computation_time`, `1.0`/`0.0` for `:is_pareto_efficient`,
  or `:net_gain`.
- `Fairness`: `1 - disparity` for the disparity metrics, the ratio for
  `:disparate_impact`; `0.0` when the required data is missing.
- `Safety`: `1.0` when safe, else `0.0` (with the same `true` defaults as
  `satisfy(::Safety, ::Dict)`).
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

Return `initial_state[:profit]`.
"""
function maximize(value::Profit, initial_state::Dict)::Float64
    profit = get(initial_state, :profit, nothing)
    isnothing(profit) && error("initial_state must contain :profit for Profit maximization.")
    return profit
end

"""
    maximize(value::Efficiency, initial_state::Dict)::Float64

Return the efficiency objective for `value.metric`: negated computation time,
`1.0`/`0.0` for Pareto efficiency, or the Kaldor-Hicks net gain.
"""
function maximize(value::Efficiency, initial_state::Dict)::Float64
    if value.metric == :computation_time
        time = get(initial_state, :computation_time, nothing)
        isnothing(time) && error("initial_state must contain :computation_time for :computation_time efficiency maximization.")
        return -time  # Negative because we want to minimize time
    elseif value.metric == :pareto
        is_pareto = get(initial_state, :is_pareto_efficient, nothing)
        isnothing(is_pareto) && error("initial_state must contain :is_pareto_efficient for :pareto efficiency maximization.")
        return is_pareto ? 1.0 : 0.0
    elseif value.metric == :kaldor_hicks
        net_gain = get(initial_state, :net_gain, nothing)
        isnothing(net_gain) && error("initial_state must contain :net_gain for :kaldor_hicks efficiency maximization.")
        return net_gain
    else
        error("Unknown efficiency metric: $(value.metric). Must be one of $(instances(EfficiencyMetric)).")
    end
end

"""
    maximize(value::Fairness, initial_state::Dict)::Float64

Return a fairness objective where higher is fairer: `1 - disparity` for the
disparity metrics and the ratio itself for `:disparate_impact`. Returns `0.0`
when the data the metric needs is missing.
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
    if value.metric == :demographic_parity
        disparity = demographic_parity(predictions, protected)
        return 1.0 - disparity  # Convert disparity to maximization score
    elseif value.metric == :equalized_odds
        isnothing(labels) && return 0.0 # Cannot compute without labels
        disparity = equalized_odds(predictions, labels, protected)
        return 1.0 - disparity
    elseif value.metric == :equal_opportunity
        isnothing(labels) && return 0.0 # Cannot compute without labels
        disparity = equal_opportunity(predictions, labels, protected)
        return 1.0 - disparity
    elseif value.metric == :disparate_impact
        return disparate_impact(predictions, protected) # DI is a ratio, 1.0 is best
    elseif value.metric == :individual_fairness
        return 1.0 - individual_fairness(predictions, similarity_matrix) # Convert individual fairness to a maximization score
    else
        error("Unknown fairness metric: $(value.metric). Please ensure it is a valid metric from FairnessMetric enum.")
    end
end

"""
    maximize(value::Safety, initial_state::Dict)::Float64

Return `1.0` when `initial_state` is safe and `0.0` otherwise, with the same
defaults as `satisfy(::Safety, ::Dict)`.
"""
function maximize(value::Safety, initial_state::Dict)::Float64
    is_safe = get(initial_state, :is_safe, true)
    invariant_holds = get(initial_state, :invariant_holds, true)
    return (is_safe && invariant_holds) ? 1.0 : 0.0
end

"""
    verify_value(value::Value, proof::Dict)::Bool

Check a caller-supplied verification *attestation* for `value` and return its
`:verified` flag.

This function does not run a prover or check a proof. It validates the shape of
the attestation — `proof[:verified]` must be present and a `Bool` — and returns
it. For a critical `Safety` value a positive attestation must also name its
`:prover`; any `:details` are logged. Producing the proof, and trusting it, is
the caller's responsibility.

When `proof` is a parsed ECHIDNA `echidna.prove.result/1` receipt (it has a
`"schema"` key), the stricter `verify_receipt` is applied instead.
"""
function verify_value(value::Value, proof::Dict)::Bool
    _is_receipt(proof) && return verify_receipt(value, proof)
    # Require verified field to be a Bool
    verified = get(proof, :verified, nothing)
    isnothing(verified) && error("proof must contain :verified field")
    isa(verified, Bool) || error("proof[:verified] must be a Bool, got $(typeof(verified))")

    return verified
end

"""
    verify_value(value::Safety, proof::Dict)::Bool

As `verify_value(::Value, ::Dict)`, and additionally require a `:prover` entry
when `value.critical` and the attestation is positive.
"""
function verify_value(value::Safety, proof::Dict)::Bool
    _is_receipt(proof) && return verify_receipt(value, proof)
    # Require verified field to be a Bool
    verified = get(proof, :verified, nothing)
    isnothing(verified) && error("proof must contain :verified field")
    isa(verified, Bool) || error("proof[:verified] must be a Bool, got $(typeof(verified))")

    # If not verified, return false immediately — no need to check prover.
    !verified && return false

    # For critical safety values that are verified, require a prover
    if value.critical
        prover = get(proof, :prover, nothing)
        isnothing(prover) && error("Critical safety proofs must contain :prover field")
        # Log prover information if details are available
        details = get(proof, :details, nothing)
        if !isnothing(details)
            @info "Safety attestation names prover: $prover" details
        end
    end

    return verified
end

"""
    PROVE_RESULT_SCHEMA

The schema identifier of the ECHIDNA prove-result receipt that
`verify_receipt` accepts: `"echidna.prove.result/1"`.
"""
const PROVE_RESULT_SCHEMA = "echidna.prove.result/1"

# The statuses the /1 schema defines; a new status needs a new schema id.
const _PROVE_STATUSES = ("verified", "failed", "error", "timeout", "unknown")

"""
    _receipt_get(receipt::AbstractDict, key::AbstractString)

Look `key` up in a parsed receipt whose keys may be `String`s (e.g. JSON.jl)
or `Symbol`s (e.g. JSON3.jl); return `nothing` when it is absent.
"""
_receipt_get(receipt::AbstractDict, key::AbstractString) =
    haskey(receipt, key) ? receipt[key] : get(receipt, Symbol(key), nothing)

"""
    _is_receipt(proof::AbstractDict)::Bool

Return whether `proof` carries a `schema` key and so should be read as a
prove-result receipt rather than a plain `:verified` attestation.
"""
_is_receipt(proof::AbstractDict)::Bool = !isnothing(_receipt_get(proof, "schema"))

"""
    _receipt_field(receipt::AbstractDict, key::AbstractString, T::Type)

Return the required receipt field `key`, throwing an `ArgumentError` when it
is missing or is not a `T`.
"""
function _receipt_field(receipt::AbstractDict, key::AbstractString, T::Type)
    v = _receipt_get(receipt, key)
    v isa T || throw(ArgumentError("$(PROVE_RESULT_SCHEMA) receipt field `$(key)` must be a $(T), got $(repr(v))"))
    return v
end

"""
    verify_receipt(value::Value, receipt::AbstractDict;
                   goal::Union{Nothing,AbstractString} = nothing,
                   allow_axioms = String[])::Bool

Check a parsed ECHIDNA `echidna.prove.result/1` receipt (the one-line JSON
object printed by `echidna prove <file> --output json`) and return whether it
establishes its goal without unaccepted assumptions.

The receipt must be well formed under the schema — `schema`, `status`
(`verified`, `failed`, `error`, `timeout` or `unknown`), `prover`, `goal` and
`trust.axioms` — or an `ArgumentError` is thrown; unknown extra fields are
ignored, as the schema requires. The result is `true` only when

- `status == "verified"` and `prover` is non-empty;
- every entry of `trust.axioms` (axioms and escape hatches such as `sorry`,
  `Admitted` or `postulate` that ECHIDNA found) is listed in `allow_axioms`;
- the receipt's `goal` equals `goal`, when `goal` is given.

Parse the JSON with any parser (`String` or `Symbol` keys both work); this
package does not run ECHIDNA. A receipt is transported evidence that a prover
checked the goal *file*: whether that file states the property `value`
expresses is the caller's claim, and passing `goal` is how that binding is made
explicit. ECHIDNA reports `trust.confidence` as `null` because it checks no
independent certificate; this function does not use it.

The receipt-versus-warrant reading follows ECHIDNA's
`docs/PROVE-RESULT-CONTRACT.adoc` and the factive/non-factive distinction of
`hyperpolymath/epistemic-types`.
"""
function verify_receipt(value::Value, receipt::AbstractDict;
                        goal::Union{Nothing,AbstractString} = nothing,
                        allow_axioms = String[])::Bool
    schema = _receipt_get(receipt, "schema")
    schema == PROVE_RESULT_SCHEMA ||
        throw(ArgumentError("not a $(PROVE_RESULT_SCHEMA) receipt (schema = $(repr(schema)))"))
    status = _receipt_field(receipt, "status", AbstractString)
    status in _PROVE_STATUSES ||
        throw(ArgumentError("status $(repr(status)) is not defined by $(PROVE_RESULT_SCHEMA)"))
    prover = _receipt_field(receipt, "prover", AbstractString)
    receipt_goal = _receipt_field(receipt, "goal", AbstractString)
    trust = _receipt_field(receipt, "trust", AbstractDict)
    axioms = _receipt_field(trust, "axioms", AbstractVector)
    all(a -> a isa AbstractString, axioms) ||
        throw(ArgumentError("$(PROVE_RESULT_SCHEMA) receipt field `trust.axioms` must hold strings"))

    status == "verified" || return false
    isempty(prover) && return false
    isnothing(goal) || receipt_goal == goal || return false
    allowed = Set{String}(String.(allow_axioms))
    return all(a -> String(a) in allowed, axioms)
end

"""
    safety_verdict(value::Safety, state::Dict)::Symbol

Return a three-valued verdict on `value` for `state`:

- `:refuted` when `state[:is_safe]` or `state[:invariant_holds]` is `false`;
- `:entailed` when both are present and `true`;
- `:unresolved` otherwise — the evidence is missing, so safety is neither
  established nor ruled out.

Unlike `satisfy(::Safety, ::Dict)`, which treats absent keys as safe, this keeps
missing evidence visible. Non-`Bool` values throw an `ArgumentError`. The verdict
names follow `hyperpolymath/ResidualEvidenceTypes.jl` (`ENTAILED` / `REFUTED` /
`UNRESOLVED`); this package does not depend on it.
"""
function safety_verdict(value::Safety, state::Dict)::Symbol
    flags = (get(state, :is_safe, nothing), get(state, :invariant_holds, nothing))
    for f in flags
        isnothing(f) || f isa Bool ||
            throw(ArgumentError("Safety evidence (:is_safe, :invariant_holds) must be Bool, got $(repr(f))"))
    end
    any(f -> f === false, flags) && return :refuted
    all(f -> f === true, flags) && return :entailed
    return :unresolved
end
