# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Fairness metric implementations.
"""

"""
    demographic_parity(predictions, protected_attributes)

Compute demographic parity disparity.

Demographic parity is satisfied when P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1) for protected attribute A.

Returns the maximum absolute difference in positive prediction rates across groups.

# Arguments
- `predictions::AbstractVector`: Binary predictions (0 or 1) or probabilities
- `protected_attributes::AbstractVector`: Protected group indicators

# Returns
- `Float64`: Maximum disparity (0 = perfect parity, 1 = maximum disparity)

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
protected = [:male, :female, :male, :female, :male, :female]
disparity = demographic_parity(predictions, protected)
```
"""
function demographic_parity(predictions::AbstractVector, protected_attributes::AbstractVector)
    @assert length(predictions) == length(protected_attributes) "Lengths must match"

    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0  # No disparity if only one group
    end

    # Compute positive rate for each group
    group_rates = Dict{Any,Float64}()
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_rates[group] = mean(group_preds)
    end

    # Maximum pairwise disparity
    rates = collect(values(group_rates))
    return maximum(rates) - minimum(rates)
end

"""
    equalized_odds(predictions, labels, protected_attributes)

Compute equalized odds disparity.

Equalized odds requires equal true positive and false positive rates across protected groups.

Returns the maximum disparity in TPR and FPR.

# Arguments
- `predictions::AbstractVector`: Binary predictions
- `labels::AbstractVector`: True labels
- `protected_attributes::AbstractVector`: Protected group indicators

# Returns
- `Float64`: Maximum disparity in error rates
"""
function equalized_odds(predictions::AbstractVector, labels::AbstractVector,
                        protected_attributes::AbstractVector)
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match"

    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end

    tpr_disparity = 0.0
    fpr_disparity = 0.0

    # Compute TPR and FPR for each group
    tprs = Float64[]
    fprs = Float64[]

    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]

        # True positives and false positives
        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fp = sum((group_preds .== 1) .& (group_labels .== 0))
        tn = sum((group_preds .== 0) .& (group_labels .== 0))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))

        # Rates
        tpr = tp > 0 || fn > 0 ? tp / (tp + fn) : 0.0
        fpr = fp > 0 || tn > 0 ? fp / (fp + tn) : 0.0

        push!(tprs, tpr)
        push!(fprs, fpr)
    end

    tpr_disparity = maximum(tprs) - minimum(tprs)
    fpr_disparity = maximum(fprs) - minimum(fprs)

    return max(tpr_disparity, fpr_disparity)
end

"""
    equal_opportunity(predictions, labels, protected_attributes)

Compute equal opportunity disparity.

Equal opportunity requires equal true positive rates across protected groups.

# Arguments
- `predictions::AbstractVector`: Binary predictions
- `labels::AbstractVector`: True labels
- `protected_attributes::AbstractVector`: Protected group indicators

# Returns
- `Float64`: Maximum TPR disparity
"""
function equal_opportunity(predictions::AbstractVector, labels::AbstractVector,
                          protected_attributes::AbstractVector)
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match"

    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end

    tprs = Float64[]

    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]

        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))

        tpr = tp > 0 || fn > 0 ? tp / (tp + fn) : 0.0
        push!(tprs, tpr)
    end

    return maximum(tprs) - minimum(tprs)
end

"""
    disparate_impact(predictions, protected_attributes)

Compute disparate impact ratio.

Disparate impact ratio is the ratio of positive prediction rates between groups.
The 80% rule suggests DI should be ≥ 0.8.

# Returns
- `Float64`: Disparate impact ratio (1.0 = no impact, <1.0 = adverse impact)
"""
function disparate_impact(predictions::AbstractVector, protected_attributes::AbstractVector)
    @assert length(predictions) == length(protected_attributes) "Lengths must match"

    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 1.0  # No impact if only one group
    end

    rates = Float64[]
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        push!(rates, mean(group_preds))
    end

    min_rate = minimum(rates)
    max_rate = maximum(rates)

    return max_rate > 0.0 ? min_rate / max_rate : 1.0
end

"""
    individual_fairness(predictions, similarity_matrix)

Compute individual fairness metric.

Individual fairness requires similar individuals to receive similar predictions.

# Arguments
- `predictions::AbstractVector`: Predictions
- `similarity_matrix::AbstractMatrix`: Pairwise similarity matrix

# Returns
- `Float64`: Average prediction difference for similar individuals
"""
function individual_fairness(predictions::AbstractVector, similarity_matrix::AbstractMatrix)
    n = length(predictions)
    @assert size(similarity_matrix) == (n, n) "Similarity matrix must be n×n"

    total_diff = 0.0
    count = 0

    for i in 1:n
        for j in (i+1):n
            if similarity_matrix[i, j] > 0.8  # Consider similar if similarity > 0.8
                total_diff += abs(predictions[i] - predictions[j])
                count += 1
            end
        end
    end

    return count > 0 ? total_diff / count : 0.0
end

"""
    satisfy(value::Fairness, state::Dict)

Check if a system state satisfies the fairness criterion.

# Arguments
- `value::Fairness`: Fairness value specification
- `state::Dict`: System state containing predictions and protected attributes

# Returns
- `Bool`: Whether the fairness criterion is satisfied

# Example

```julia
fairness = Fairness(metric = :demographic_parity, threshold = 0.05)
state = Dict(
    :predictions => [1, 0, 1, 1],
    :protected => [:male, :female, :male, :female]
)
satisfy(fairness, state)  # true if disparity < 0.05
```
"""
function satisfy(value::Fairness, state::Dict)
    # Extract data from state
    predictions = get(state, :predictions, nothing)
    protected = get(state, :protected, get(state, :protected_attributes, nothing))
    labels = get(state, :labels, nothing)

    isnothing(predictions) && error("State must contain :predictions")
    isnothing(protected) && error("State must contain :protected or :protected_attributes")

    # Compute disparity based on metric
    disparity = if value.metric == :demographic_parity
        demographic_parity(predictions, protected)
    elseif value.metric == :equalized_odds
        isnothing(labels) && error("equalized_odds requires :labels in state")
        equalized_odds(predictions, labels, protected)
    elseif value.metric == :equal_opportunity
        isnothing(labels) && error("equal_opportunity requires :labels in state")
        equal_opportunity(predictions, labels, protected)
    elseif value.metric == :disparate_impact
        di = disparate_impact(predictions, protected)
        return di >= 0.8  # 80% rule
    else
        error("Unknown fairness metric: $(value.metric)")
    end

    # Check against threshold
    return disparity <= value.threshold
end
