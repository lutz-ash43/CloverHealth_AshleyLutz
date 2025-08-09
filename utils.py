import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor (VIF) for a set of features.

    Parameters:
    - df: pandas DataFrame containing the data
    - features: list of column names to assess

    Returns:
    - vif_df: DataFrame with features and their VIF scores
    """
    X = df[features].copy()
    # If categorical, get dummies
    X = pd.get_dummies(X, drop_first=True)
    vif_data = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": X.columns[i], "VIF": vif})
    return pd.DataFrame(vif_data) 

def generate_later_timestamp_column(df, reference_col, new_col='later_timestamp', min_days=1, max_days=30, seed=None):
    """
    Adds a new timestamp column where each value is after the corresponding value in `reference_col`.

    Parameters:
    - df: pandas DataFrame
    - reference_col: name of the column with base timestamps
    - new_col: name of the new column to create
    - min_days: minimum number of days after the reference date
    - max_days: maximum number of days after the reference date
    - seed: random seed for reproducibility

    Returns:
    - df with the new column added
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure the reference column is in datetime format
    df = df.copy()
    df[reference_col] = pd.to_datetime(df[reference_col])

    # Generate random number of days to add
    days_to_add = np.random.randint(min_days, max_days + 1, size=len(df))
    df[new_col] = df[reference_col] + pd.to_timedelta(days_to_add, unit='D')

    return df

def plot_outcome_by_physician_hra(data): 
    data_piv = data.fillna("pass")
    data_piv["outcome"] = data_piv.outcome.replace({"pass": 0, "failure": 1})
    data_piv = data_piv.pivot_table(
        index='servicing_provider_id',  # or name
        columns='health_risk_assesment',
        values='outcome',
        aggfunc='mean'
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(data_piv, cmap="coolwarm", annot=False, fmt=".2f", cbar_kws={'label': 'Failure Rate'})
    plt.title("Failure Rate by Provider and Health Risk Score")
    plt.xlabel("Health Risk Assessment")
    plt.ylabel("Provider")
    plt.tight_layout()
    plt.show()

def plot_failure_rates(data_outcome_compare, bad_providers): 
    provider_counts = data_outcome_compare.groupby("servicing_provider_id")["outcome"].count()
    provider_failures = data_outcome_compare[data_outcome_compare["outcome"] == "failure"].groupby("servicing_provider_id")["outcome"].count()
    failure_rate = (provider_failures / provider_counts).fillna(0)
    result = pd.DataFrame({
        "failure_rate": failure_rate,
        "count": provider_counts
    }).sort_values("failure_rate", ascending=False)

    # Plot distribution of failure rates
    plt.figure(figsize=(8, 5))
    sns.histplot(result["failure_rate"], bins=20, kde=True, color="blue")
    plt.xlabel("Provider Failure Rate")
    plt.ylabel("Number of Providers")
    plt.title("Distribution of Provider Failure Rates")

    # Add vertical lines for significantly worse providers
    for pid in bad_providers:
        if pid in result.index:
            plt.axvline(result.loc[pid, "failure_rate"], color="red", linestyle="--", label=f"Provider {pid}")
    if bad_providers:
        plt.legend()
    plt.show()
    
    return result.sort_index()

def compare_covariate_distributions_by_provider_quality(data_outcome_compare, bad_providers, covariates):
    """
    Compare distributions of health risk assessment scores for providers with high failure rates vs others.
    
    Parameters:
    - data_outcome_compare: DataFrame containing the outcome data.
    - bad_providers: List of provider IDs with high failure rates.
    
    Returns:
    - ks_stat: KS statistic
    - p_value: p-value from the KS test
    """
    data_outcome_compare['provider_group'] = data_outcome_compare['servicing_provider_id'].apply(lambda x: 'poor' if x in bad_providers else 'other')

    kde_data = []
    for covariate in covariates:
        group_poor = data_outcome_compare[data_outcome_compare['provider_group'] == 'poor'][covariate]
        group_other = data_outcome_compare[data_outcome_compare['provider_group'] == 'other'][covariate]
        ks_stat, p_value = ks_2samp(group_poor, group_other)
        kde_data.append((covariate, group_poor, group_other, ks_stat, p_value))
        print(f"KS statistic for {covariate}: {ks_stat:.4f}, p-value: {p_value:.4f}")

    
    fig, axes = plt.subplots(1, len(covariates), figsize=(7 * len(covariates), 5))
    if len(covariates) == 1:
        axes = [axes]
    for ax, (covariate, group_poor, group_other, ks_stat, p_value) in zip(axes, kde_data):
        sns.kdeplot(group_poor, label='Poor Providers', fill=True, alpha=0.5, color='red', ax=ax)
        sns.kdeplot(group_other, label='Other Providers', fill=True, alpha=0.5, color='blue', ax=ax)
        ax.set_title(f'Distribution of {covariate}\nKS stat: {ks_stat:.4f}, p-value: {p_value:.4f}')
        ax.set_xlabel(covariate)
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    plt.show()

def cauchy_pvalue_combination(pvalues):
    """
    Combine p-values using the Cauchy combination test.

    Parameters:
    - pvalues: array-like, list or numpy array of p-values (should be between 0 and 1)

    Returns:
    - combined_pvalue: float, the combined p-value
    """
    pvalues = np.asarray(pvalues)
    # Avoid p-values exactly 0 or 1 for numerical stability
    pvalues = np.clip(pvalues, 1e-15, 1 - 1e-15)
    t = np.tan((0.5 - pvalues) * np.pi)
    cauchy_stat = np.mean(t)
    combined_pvalue = 1 - (np.arctan(cauchy_stat) / np.pi + 0.5)
    return combined_pvalue