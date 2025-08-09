import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def empirical_null_provider_failures(df, target_col, provider_col, covariates, n_simulations=1000, random_state=42):
    """
    Compare observed provider failure rates to an empirical null conditioned on covariates.

    Parameters:
    - df: DataFrame with data
    - target_col: column name for binary outcome (0/1)
    - provider_col: column name for provider/grouping
    - covariates: list of covariates to condition on
    - n_simulations: number of simulations for null distribution

    Returns:
    - results_df: DataFrame with observed, expected mean, p-value, and CI per provider
    """
    df = df.copy()
    df = df.dropna(subset=[target_col] + covariates + [provider_col])
    # binarize target column 
    df[target_col] = df[target_col].apply(lambda x: 1 if x == 'failure' else 0)

    provider_results = []

    for provider, group in tqdm(df.groupby(provider_col), desc="Simulating nulls by provider"):
        n = len(group)
        # removing 1 and 10 from empirical null calcualtion since these have only 1 gender each
        train_df = df[(df[provider_col] != provider)]
        test_df = group.copy()
        
        X_train = train_df[covariates]
        y_train = train_df[target_col]

        model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)
        # true failure rate per provider
        observed_fail_rate = group[target_col].mean()
        predicted_probs = model.predict_proba(test_df[covariates])[:, 1]

        # Simulate n_simulations provider-level failure rates
        sim_fail_rates = []
        for _ in range(n_simulations):
            # simulate outcomes based on predicted probabilities for that providers patients based only on covariates
            simulated_outcomes = np.random.binomial(n=1, p=predicted_probs)
            sim_fail_rates.append(simulated_outcomes.mean())

        sim_fail_rates = np.array(sim_fail_rates)
        # determine p-value and confidence intervals
        lower = np.percentile(sim_fail_rates, 2.5)
        upper = np.percentile(sim_fail_rates, 97.5)
        p_value = (np.sum(sim_fail_rates >= observed_fail_rate) + 1) / (n_simulations + 1) if observed_fail_rate > sim_fail_rates.mean() else (np.sum(sim_fail_rates <= observed_fail_rate) + 1) / (n_simulations + 1)

        provider_results.append({
            "provider": provider,
            "observed_failure_rate": observed_fail_rate,
            "expected_mean": sim_fail_rates.mean(),
            "lower_95_CI": lower,
            "upper_95_CI": upper,
            "p_value": p_value
        })

    results_df = pd.DataFrame(provider_results)
    return results_df

def bootstrap_stability_check(df, target_col, provider_col, covariates, n_bootstraps=10):
    boostrap_results = []
    for _ in range(n_bootstraps):
        boot_df = df.sample(frac=1, replace=True)
        res = empirical_null_provider_failures(boot_df, target_col, provider_col, covariates)
        boostrap_results.append(res)
    return(pd.concat(boostrap_results, ignore_index=True))