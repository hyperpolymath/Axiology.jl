# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

"""
    demographic_parity(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64

Return the demographic-parity disparity: the largest difference in mean
prediction (the positive-prediction rate, for binary predictions) between any
two protected groups. `0.0` means perfect parity.

Returns `0.0` when fewer than two groups are present.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
protected = [:A, :A, :B, :B, :B, :A]
demographic_parity(predictions, protected)  # 0.0 (both groups have rate 2/3)
```
"""
function demographic_parity(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    group_rates = Dict{Any,Float64}()
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_rates[group] = isempty(group_preds) ? 0.0 : mean(group_preds)
    end
    rates = collect(values(group_rates))
    return maximum(rates) - minimum(rates)
end

"""
    equalized_odds(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                   protected_attributes::AbstractVector)::Float64

Return the equalized-odds disparity: the larger of the maximum between-group
difference in true-positive rate (TPR) and in false-positive rate (FPR).
`0.0` means the groups are treated alike on both rates.

A group with no positive labels has no defined TPR and is left out of the TPR
comparison; likewise a group with no negative labels is left out of the FPR
comparison. Each comparison needs at least two groups with a defined rate and
otherwise contributes `0.0`. Returns `0.0` when fewer than two groups are present.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
labels      = [1, 0, 0, 1, 0, 1]
protected   = [:A, :A, :B, :B, :B, :A]
equalized_odds(predictions, labels, protected)
```
"""
function equalized_odds(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                        protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    # Only include groups that have positives (for TPR) or negatives (for FPR).
    # Groups with no positives have undefined TPR; including them as 0.0 creates
    # artificial disparity when other groups achieve TPR=1.0 on their (all-positive) data.
    tprs = Float64[]
    fprs = Float64[]
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fp = sum((group_preds .== 1) .& (group_labels .== 0))
        tn = sum((group_preds .== 0) .& (group_labels .== 0))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))
        # Only count TPR for groups that have at least one positive example
        if (tp + fn) > 0
            push!(tprs, tp / (tp + fn))
        end
        # Only count FPR for groups that have at least one negative example
        if (fp + tn) > 0
            push!(fprs, fp / (fp + tn))
        end
    end
    max_tpr_disparity = length(tprs) >= 2 ? maximum(tprs) - minimum(tprs) : 0.0
    max_fpr_disparity = length(fprs) >= 2 ? maximum(fprs) - minimum(fprs) : 0.0
    return max(max_tpr_disparity, max_fpr_disparity)
end

"""
    equal_opportunity(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                      protected_attributes::AbstractVector)::Float64

Return the equal-opportunity disparity: the maximum between-group difference in
true-positive rate (TPR) only. `0.0` means individuals who merit a positive
outcome receive one at the same rate in every group.

Groups with no positive labels have no defined TPR and are left out; if fewer
than two groups have a defined TPR the result is `0.0`.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
labels      = [1, 0, 0, 1, 0, 1]
protected   = [:A, :A, :B, :B, :B, :A]
equal_opportunity(predictions, labels, protected)
```
"""
function equal_opportunity(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                          protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    # Only include groups that have at least one positive label.
    # Groups with no positives have undefined TPR; including them as 0.0 creates
    # artificial disparity when other groups achieve TPR=1.0 on their (all-positive) data.
    tprs = Float64[]
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))
        if (tp + fn) > 0
            push!(tprs, tp / (tp + fn))
        end
    end
    return length(tprs) >= 2 ? maximum(tprs) - minimum(tprs) : 0.0
end

"""
    disparate_impact(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64

Return the disparate-impact ratio: the lowest group selection rate divided by
the highest. `1.0` means equal selection rates; values near `0.0` mean strong
disparity. Under the "four-fifths rule" a ratio below `0.8` is commonly read as
evidence of adverse impact.

Returns `1.0` when fewer than two groups are present or when no group is ever
selected.

Note that this is a ratio where *higher is fairer*, the opposite orientation to
the disparity metrics above; `satisfy(::Fairness, …)` therefore reads a
`:disparate_impact` threshold as a minimum ratio (e.g. `threshold = 0.8`).

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
protected   = [:A, :A, :B, :B, :B, :A]
disparate_impact(predictions, protected)  # 1.0
```
"""
function disparate_impact(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 1.0
    end
    group_rates = Dict{Any,Float64}()
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_rates[group] = isempty(group_preds) ? 0.0 : mean(group_preds)
    end
    rates = collect(values(group_rates))
    min_rate = minimum(rates)
    max_rate = maximum(rates)
    return max_rate > 0.0 ? min_rate / max_rate : 1.0
end

"""
    individual_fairness(predictions::AbstractVector, similarity_matrix::AbstractMatrix;
                        similarity_threshold::Float64 = 0.8)::Float64

Return the mean absolute prediction difference over all pairs of individuals
whose similarity exceeds `similarity_threshold` ("similar individuals should be
treated similarly"). `0.0` is best.

`similarity_matrix` must be `n×n` for `n` predictions, with entries normally in
`[0, 1]`. Returns `0.0` when no pair is similar enough to compare.

# Example

```julia
predictions = [0.8, 0.2, 0.7, 0.3]
similarity  = [1.0 0.9 0.1 0.2;
               0.9 1.0 0.2 0.1;
               0.1 0.2 1.0 0.85;
               0.2 0.1 0.85 1.0]
individual_fairness(predictions, similarity; similarity_threshold = 0.8)
```
"""
function individual_fairness(predictions::AbstractVector, similarity_matrix::AbstractMatrix;
                            similarity_threshold::Float64 = 0.8)::Float64
    n = length(predictions)
    @assert size(similarity_matrix) == (n, n) "Similarity matrix must be n×n."
    @assert similarity_threshold >= 0.0 && similarity_threshold <= 1.0 "Threshold must be in [0, 1]."
    total_diff = 0.0
    count = 0
    for i in 1:n
        for j in (i+1):n
            if similarity_matrix[i, j] > similarity_threshold
                total_diff += abs(predictions[i] - predictions[j])
                count += 1
            end
        end
    end
    return count > 0 ? total_diff / count : 0.0
end

"""
    satisfy(value::Fairness, state::Dict)::Bool

Return whether `state` meets the fairness criterion `value`.

`state` must hold `:predictions`, plus `:protected` (or `:protected_attributes`)
for the group metrics, `:labels` for `:equalized_odds`/`:equal_opportunity`, and
`:similarity_matrix` for `:individual_fairness`. A missing key raises an error.

For the disparity metrics the check is `disparity <= value.threshold`. For
`:disparate_impact` it is `ratio >= value.threshold`, so the threshold is a
minimum ratio there (the default `0.05` is far looser than the usual `0.8`).
"""
function satisfy(value::Fairness, state::Dict)::Bool
    predictions = get(state, :predictions, nothing)
    protected = get(state, :protected, get(state, :protected_attributes, nothing))
    labels = get(state, :labels, nothing)
    similarity_matrix = get(state, :similarity_matrix, nothing)

    isnothing(predictions) && error("State must contain :predictions for fairness evaluation.")
    isnothing(protected) && value.metric != :individual_fairness && error("State must contain :protected for group fairness.")
    isnothing(similarity_matrix) && value.metric == :individual_fairness && error("State must contain :similarity_matrix for individual fairness.")

    disparity = if value.metric == :demographic_parity
        demographic_parity(predictions, protected)
    elseif value.metric == :equalized_odds
        isnothing(labels) && error("equalized_odds metric requires :labels in state.")
        equalized_odds(predictions, labels, protected)
    elseif value.metric == :equal_opportunity
        isnothing(labels) && error("equal_opportunity metric requires :labels in state.")
        equal_opportunity(predictions, labels, protected)
    elseif value.metric == :disparate_impact
        di_ratio = disparate_impact(predictions, protected)
        return di_ratio >= value.threshold
    elseif value.metric == :individual_fairness
        individual_fairness(predictions, similarity_matrix)
    else
        error("Unknown fairness metric: $(value.metric).")
    end

    return disparity <= value.threshold
end
