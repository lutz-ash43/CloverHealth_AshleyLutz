import pandas as pd 
import numpy as np
import pymc as pm
from lifelines import CoxPHFitter
from patsy import dmatrix

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

# lets create a survival analysis to determine the effect of provider and covariates on time to failure

def cox_provider_effect(
    df, 
    start_col,
    end_col, 
    event_col, 
    provider_col, 
    covariates=['member_sex', 'health_risk_assesment']
):
    """
    Fits a Cox proportional hazards model with provider as a factor and specified covariates.

    Parameters:
    - df: pandas DataFrame with survival data
    - duration_col: name of the time-to-event column
    - event_col: name of the event indicator column (1=event, 0=censor)
    - provider_col: name of the provider categorical column
    - covariates: list of covariate column names to adjust for

    Returns:
    - model: fitted lifelines CoxPHFitter object
    - summary_df: DataFrame with model summary including provider hazard ratios and covariates
    """
    df = df.copy()
    df[event_col] = df[event_col].apply(lambda x: 1 if x == 'failure' else 0)
    df["duration"] = df[end_col] - df[start_col]
    df["duration"] = df["duration"].dt.days  # Convert timedelta to days
    
    # Convert provider to categorical and create dummies (one-hot encode, drop one to avoid collinearity)
    provider_dummies = pd.get_dummies(df[provider_col], prefix='provider', drop_first=True)

    # Combine covariates
    X = pd.concat([provider_dummies, df[covariates]], axis=1)

    # Combine with duration and event columns
    survival_df = pd.concat([df[["duration", event_col]], X], axis=1)

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col="duration", event_col=event_col)

    return cph, cph.summary

def bayesian_cox_model(
    df,
    start_col,
    end_col,
    event_col,
    provider_col,
    covariates=["member_sex", "health_risk_assesment"],
    draws=1000,
    tune=1000,
    chains=4,
):
    """
    Fit a Bayesian Cox-style survival model using exponential hazards with right-censoring.
    """

    df = df.copy()

    # Binary outcome: 1 = failure, 0 = censored
    df[event_col] = df[event_col].apply(lambda x: 1 if x == "failure" else 0)

    # Duration in days
    df["duration"] = (df[end_col] - df[start_col]).dt.days
    df = df.dropna(subset=["duration", event_col] + covariates)

    # Encode providers
    provider_codes, provider_names = pd.factorize(df[provider_col])
    n_providers = len(provider_names)

    # Design matrix
    X = dmatrix("0 + " + " + ".join(covariates), df, return_type="dataframe")
    X = (X - X.mean()) / X.std()

    duration = df["duration"].values
    event = df[event_col].values
    provider_codes = np.asarray(provider_codes)

    with pm.Model() as model:
        # Priors
        log_h0 = pm.Normal("log_h0", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        sigma_provider = pm.HalfNormal("sigma_provider", sigma=1)
        provider_offset = pm.Normal("provider_offset", mu=0, sigma=1, shape=n_providers)
        provider_effect = pm.Deterministic("provider_effect", provider_offset * sigma_provider)

        # Linear predictor and hazard
        linear_pred = pm.math.dot(X.values, beta) + provider_effect[provider_codes]
        hazard = pm.math.exp(log_h0 + linear_pred)

        # Custom log-likelihood for exponential survival model with censoring
        def logp_exp_survival(t, event, hazard):
            # log-likelihood:
            # for events:     log(h) - h * t
            # for censored:   - h * t
            loglik = event * pm.math.log(hazard) - hazard * t
            return pm.math.sum(loglik)

        # Use a custom distribution to define the likelihood
        pm.Potential("likelihood", logp_exp_survival(duration, event, hazard))

        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9, return_inferencedata=True)

    return model, idata, provider_names

def plot_bayesian_cox_effects(idata, provider_names, covariate_names, hdi_prob=0.95):
    import matplotlib.pyplot as plt
    import arviz as az

    # PROVIDER EFFECTS
    provider_effects = idata.posterior["provider_effect"]
    provider_hdi = az.hdi(provider_effects, hdi_prob=hdi_prob)
    provider_means = provider_effects.mean(dim=["chain", "draw"])

    provider_dim = list(provider_means.dims)[0]  

    plt.figure(figsize=(10, len(provider_names) * 0.3 + 3))
    for i, name in enumerate(provider_names):
        mean_val = provider_means.sel({provider_dim: i}).values.item()
        lower = provider_hdi.sel({provider_dim: i, "hdi": "lower"})["provider_effect"]
        upper = provider_hdi.sel({provider_dim: i, "hdi": "higher"})["provider_effect"]
        plt.errorbar(mean_val, i, xerr=[[mean_val - lower], [upper - mean_val]], fmt="o", color="black")

    plt.axvline(0, linestyle="--", color="red", alpha=0.5)
    plt.yticks(range(len(provider_names)), provider_names)
    plt.title("Provider Effects (Posterior Mean ± 95% HDI)")
    plt.xlabel("Effect on log hazard")
    plt.tight_layout()
    plt.show()

    # COVARIATE EFFECTS
    beta_effects = idata.posterior["beta"]
    beta_means = beta_effects.mean(dim=["chain", "draw"])
    beta_hdi = az.hdi(beta_effects, hdi_prob=hdi_prob)
    beta_dim = list(beta_means.dims)[0]

    plt.figure(figsize=(8, len(covariate_names) * 0.6 + 2))
    for i, cov in enumerate(covariate_names):
        mean_val = beta_means.sel({beta_dim: i}).values.item()
        lower = beta_hdi.sel({beta_dim: i, "hdi": "lower"})["beta"]
        upper = beta_hdi.sel({beta_dim: i, "hdi": "higher"})["beta"]
        plt.errorbar(mean_val, i, xerr=[[mean_val - lower], [upper - mean_val]], fmt="o", color="blue")

    plt.axvline(0, linestyle="--", color="red", alpha=0.5)
    plt.yticks(range(len(covariate_names)), covariate_names)
    plt.title("Covariate Effects (Posterior Mean ± 95% HDI)")
    plt.xlabel("Effect on log hazard")
    plt.tight_layout()
    plt.show()
