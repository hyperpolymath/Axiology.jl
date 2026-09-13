# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

"""
    demographic_parity(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64

Compute the demographic parity disparity across protected groups.

Demographic parity requires that the positive prediction rate (the proportion of
individuals receiving a favorable outcome) is equal across all protected groups.
This function calculates the maximum difference in positive prediction rates
between any two groups, with a value of 0.0 indicating perfect parity.

# Arguments
- `predictions::AbstractVector`: Binary predictions or scores for each individual.
- `protected_attributes::AbstractVector`: Group membership for each individual
                                         (e.g., gender, race).

# Returns
- `Float64`: The maximum difference in positive prediction rates between groups.
             Returns 0.0 if there are fewer than 2 groups.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
protected = [:A, :A, :B, :B, :B, :A]
disparity = demographic_parity(predictions, protected)
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

Compute the equalized odds disparity across protected groups.

Equalized odds requires that both the true positive rate (TPR) and false positive
rate (FPR) are equal across all protected groups. This function calculates the
maximum disparity in either TPR or FPR between groups, with a value of 0.0
indicating perfect equalized odds.

# Arguments
- `predictions::AbstractVector{<:Real}`: Binary predictions (typically 0 or 1) for
                                         each individual.
- `labels::AbstractVector{<:Real}`: True binary labels (typically 0 or 1) for each
                                    individual.
- `protected_attributes::AbstractVector`: Group membership for each individual
                                         (e.g., gender, race).

# Returns
- `Float64`: The maximum disparity in TPR or FPR between groups. Returns 0.0 if
             there are fewer than 2 groups.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
labels = [1, 0, 0, 1, 0, 1]
protected = [:A, :A, :B, :B, :B, :A]
disparity = equalized_odds(predictions, labels, protected)
```
"""
function equalized_odds(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                        protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    tpr_disparities = Float64[]
    fpr_disparities = Float64[]
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fp = sum((group_preds .== 1) .& (group_labels .== 0))
        tn = sum((group_preds .== 0) .& (group_labels .== 0))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))
        tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0.0
        push!(tpr_disparities, tpr)
        push!(fpr_disparities, fpr)
    end
    max_tpr_disparity = maximum(tpr_disparities) - minimum(tpr_disparities)
    max_fpr_disparity = maximum(fpr_disparities) - minimum(fpr_disparities)
    return max(max_tpr_disparity, max_fpr_disparity)
end

"""
    equal_opportunity(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                     protected_attributes::AbstractVector)::Float64

Compute the equal opportunity disparity across protected groups.

Equal opportunity is a weaker form of equalized odds that focuses only on the
true positive rate (TPR). It requires that individuals who truly deserve a
positive outcome have an equal chance of receiving it, regardless of their
protected group membership. This function calculates the maximum difference in
TPR between groups, with a value of 0.0 indicating perfect equal opportunity.

# Arguments
- `predictions::AbstractVector{<:Real}`: Binary predictions (typically 0 or 1) for
                                         each individual.
- `labels::AbstractVector{<:Real}`: True binary labels (typically 0 or 1) for each
                                    individual.
- `protected_attributes::AbstractVector`: Group membership for each individual
                                         (e.g., gender, race).

# Returns
- `Float64`: The maximum difference in TPR between groups. Returns 0.0 if there
             are fewer than 2 groups.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
labels = [1, 0, 0, 1, 0, 1]
protected = [:A, :A, :B, :B, :B, :A]
disparity = equal_opportunity(predictions, labels, protected)
```
"""
function equal_opportunity(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                          protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match."
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
        tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        push!(tprs, tpr)
    end
    return maximum(tprs) - minimum(tprs)
end

"""
    disparate_impact(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64

Compute the disparate impact ratio across protected groups.

Disparate impact measures whether the selection rate (positive prediction rate)
for a protected group is substantially less than for other groups. This function
returns the ratio of the minimum selection rate to the maximum selection rate
across all groups. A value of 1.0 indicates no disparate impact, while values
closer to 0.0 indicate greater disparity. The "80% rule" commonly used in hiring
suggests that a ratio below 0.8 may indicate adverse impact.

# Arguments
- `predictions::AbstractVector`: Binary predictions or scores for each individual.
- `protected_attributes::AbstractVector`: Group membership for each individual
                                         (e.g., gender, race).

# Returns
- `Float64`: The ratio of minimum to maximum selection rates across groups.
             Returns 1.0 if there are fewer than 2 groups or if the maximum rate is 0.0.

# Example

```julia
predictions = [1, 0, 1, 1, 0, 1]
protected = [:A, :A, :B, :B, :B, :A]
ratio = disparate_impact(predictions, protected)
# A ratio < 0.8 may indicate disparate impact under the 80% rule
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

Compute the individual fairness metric based on similarity between individuals.

Individual fairness requires that similar individuals receive similar treatment.
This function measures the average absolute difference in predictions between
pairs of individuals whose similarity exceeds a given threshold. Lower values
indicate better individual fairness (more similar predictions for similar
individuals).

# Arguments
- `predictions::AbstractVector`: Predictions or scores for each individual.
- `similarity_matrix::AbstractMatrix`: An n×n matrix where `similarity_matrix[i, j]`
                                       indicates the similarity between individuals i and j.
                                       Values should typically be in [0, 1].
- `similarity_threshold::Float64`: The minimum similarity required for two individuals
                                   to be considered "similar" and compared. Defaults to 0.8.

# Returns
- `Float64`: The average absolute difference in predictions between similar individuals.
             Returns 0.0 if no pairs of individuals exceed the similarity threshold.

# Example

```julia
predictions = [0.8, 0.2, 0.7, 0.3]
similarity = [1.0 0.9 0.1 0.2;
              0.9 1.0 0.2 0.1;
              0.1 0.2 1.0 0.85;
              0.2 0.1 0.85 1.0]
fairness = individual_fairness(predictions, similarity, similarity_threshold=0.85)
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
