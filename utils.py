import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

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