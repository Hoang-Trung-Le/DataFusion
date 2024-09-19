import numpy as np
import pandas as pd


def z_score_threshold(probabilities_df, threshold=-1.96):
    # Calculate mean and standard deviation for each segment
    mean_probs = probabilities_df.mean(axis=1)
    std_probs = probabilities_df.std(axis=1)

    # Calculate z-score for each model's probability
    z_scores = (probabilities_df.subtract(mean_probs, axis=0)).divide(std_probs, axis=0)

    # Eliminate models with z-score below threshold
    eliminated_models = (
        z_scores[z_scores < threshold].dropna(axis=1, how="all").columns.tolist()
    )

    return eliminated_models, z_scores


# Example usage
# Assuming 'dsi_ensemble' is a dataframe where each column is a model's probability for a segment
# eliminated_models, z_scores = z_score_threshold(dsi_ensemble, threshold=-1.96)
# print("Models eliminated based on z-score:", eliminated_models)


from scipy import stats


def t_test_threshold(probabilities_df, expected_mean=1 / 6, alpha=0.05):
    eliminated_models = []

    for model in probabilities_df.columns:
        t_stat, p_value = stats.ttest_1samp(probabilities_df[model], expected_mean)
        if p_value < alpha and probabilities_df[model].mean() < expected_mean:
            eliminated_models.append(model)

    return eliminated_models


# Example usage
# Assuming 'dsi_ensemble' is a dataframe where each column is a model's probability for a segment
# eliminated_models = t_test_threshold(dsi_ensemble, expected_mean=1/6, alpha=0.05)
# print("Models eliminated based on t-test:", eliminated_models)


import math


def entropy_threshold(probabilities_df, entropy_factor=1.5):
    def entropy(probabilities):
        return -np.sum([p * math.log(p) for p in probabilities if p > 0])

    # Calculate entropy for each segment
    entropy_values = probabilities_df.apply(entropy, axis=1)

    # Calculate mean entropy
    mean_entropy = entropy_values.mean()

    # Eliminate models that contribute to entropy more than allowed
    eliminated_models = []
    for model in probabilities_df.columns:
        model_entropy = probabilities_df[model].apply(
            lambda p: -p * math.log(p) if p > 0 else 0
        )
        if model_entropy.mean() > entropy_factor * mean_entropy:
            eliminated_models.append(model)

    return eliminated_models, entropy_values


# Example usage
# eliminated_models, entropy_values = entropy_threshold(dsi_ensemble, entropy_factor=1.5)
# print("Models eliminated based on entropy:", eliminated_models)
