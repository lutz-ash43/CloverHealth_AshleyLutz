import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import fisher_exact
from tqdm import tqdm

def provider_propensity_matching(
    df,
    covariates,
    outcome_col,
    provider_col,
    id_col=None,
    n_neighbors=3,
    caliper=0.05,
    min_sample_size=10,
    standardize=True,
):
    """
    Performs provider-level propensity score matching to estimate the effect of providers on a binary outcome.

    For each provider, the function matches treated (provider's patients) and control (other providers' patients)
    using propensity scores calculated from specified covariates. Nearest neighbor matching within a caliper is used,
    and Fisher's exact test is performed on the matched sample to assess the association between provider and outcome.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing patient-level data.
    covariates : list of str
        List of column names to use as covariates for propensity score modeling.
    outcome_col : str
        Name of the binary outcome column.
    provider_col : str
        Name of the column indicating provider assignment.
    id_col : str or None, optional
        Name of the unique identifier column for each row (default is None).
    n_neighbors : int, optional
        Number of nearest neighbors to match for each treated sample (default is 3).
    caliper : float, optional
        Maximum allowed difference in propensity scores for matching (default is 0.05).
    min_sample_size : int, optional
        Minimum number of treated samples required for a provider to be evaluated (default is 10).
    standardize : bool, optional
        Whether to standardize covariates before propensity score modeling (default is True).

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing results for each provider, including:
            - provider: Provider identifier
            - n_treated: Number of treated samples
            - n_matched: Number of matched pairs
            - odds_ratio: Odds ratio from Fisher's exact test
            - p_value: p-value from Fisher's exact test
            - outcome_table: Outcome contingency table (as dict)
    """
    results = []
    providers = df[provider_col].unique()

    for provider in tqdm(providers, desc="Evaluating providers"):
        df_provider = df[df[provider_col] == provider].copy()
        # need to subsample the other providers to avoid biasing the propensity score model
        df_others = df[df[provider_col] != provider].copy().sample(300)

        if len(df_provider) < min_sample_size:
            continue

        # Combine for propensity score model
        df_provider['group'] = 1
        df_others['group'] = 0
        df_all = pd.concat([df_provider, df_others])

        # Optional: standardize covariates
        X = df_all[covariates].copy()

        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Propensity score model
        ps_model = LogisticRegression(class_weight='balanced', max_iter=1000)
        ps_model.fit(X, df_all["group"])
        df_all["propensity_score"] = ps_model.predict_proba(X)[:, 1]

        # Split again with scores
        treated = df_all[df_all["group"] == 1]
        control = df_all[df_all["group"] == 0]

        # Nearest neighbor matching with caliper
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(control[["propensity_score"]])
        distances, indices = nn.kneighbors(treated[["propensity_score"]])

        matches = []
        for i, (dist, idxs) in enumerate(zip(distances, indices)):
            if dist[0] <= caliper:
                matches.append((treated.index[i], control.index[idxs[0]]))

        if not matches:
            continue

        # Create matched dataframe
        matched_rows = []
        for treat_idx, control_idx in matches:
            matched_rows.append(df_all.loc[treat_idx])
            matched_rows.append(df_all.loc[control_idx])

        df_matched = pd.DataFrame(matched_rows)

        # Count outcomes
        outcome_table = pd.crosstab(df_matched["group"], df_matched[outcome_col])

        # Skip if table is malformed
        if outcome_table.shape != (2, 2):
            continue

        # Fisher's exact test
        oddsratio, p_value = fisher_exact(outcome_table, alternative='greater')

        results.append({
            "provider": provider,
            "n_treated": len(treated),
            "n_matched": len(matched_rows) // 2,
            "odds_ratio": oddsratio,
            "p_value": p_value,
            "outcome_table": outcome_table.to_dict()
        })

    return pd.DataFrame(results)
