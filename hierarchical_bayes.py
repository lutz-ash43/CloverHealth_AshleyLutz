import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler, LabelEncoder



def fit_hierarchical_logistic_model(df, covariates, target_col, provider_col, standardize=True, draws=2000, tune=1000, chains=4):
    """
    Fits a Bayesian hierarchical logistic regression model using PyMC.

    Parameters:
    - df: pandas DataFrame with all data
    - covariates: list of column names to use as fixed effects
    - target_col: name of the binary outcome column (0/1)
    - provider_col: name of the provider/grouping column
    - standardize: whether to standardize covariates
    - draws, tune, chains: MCMC settings

    Returns:
    - idata: ArviZ InferenceData object
    """
    df = df.copy()

    # Encode provider as integer
    provider_encoder = LabelEncoder()
    df["provider_idx"] = provider_encoder.fit_transform(df[provider_col])
    n_providers = df["provider_idx"].nunique()

    # binarize the target column if not already binary
    if df[target_col].dtype != 'int' and df[target_col].dtype != 'bool':
        df[target_col] = df[target_col].apply(lambda x: 1 if x == 'failure' else 0)
    

    # Optionally standardize covariates
    if standardize:
        scaler = StandardScaler()
        df[covariates] = scaler.fit_transform(df[covariates])

    X = df[covariates].values
    y = df[target_col].values
    provider_idx = df["provider_idx"].values

    with pm.Model() as model:
        # Hyperprior for provider random effects
        sigma_provider = pm.HalfNormal("sigma_provider", sigma=1.0)

        # Random intercepts for each provider
        provider_offset = pm.Normal("provider_offset", mu=0.0, sigma=1.0, shape=n_providers)
        provider_effect = pm.Deterministic("mu_provider", provider_offset * sigma_provider)

        # Intercept
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.5)

        # Fixed effect coefficients
        betas = pm.Normal("betas", mu=0.0, sigma=0.5, shape=X.shape[1])

        # Linear model
        mu = intercept + pm.math.dot(X, betas) + provider_effect[provider_idx]

        # Likelihood (logistic)
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
        print(y)
        y_obs = pm.Bernoulli("y_obs", p=theta, observed=y)
        
        # Sampling
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9, return_inferencedata=True)

    return idata, provider_encoder

def plot_bayesian_coefficients(idata, provider_encoder, covariates, hdi_prob=0.95):
    """
    Plots posterior summaries of provider random effects and fixed effect coefficients,
    using a consistent layout like `plot_bayesian_cox_effects`.
    """
    posterior = idata.posterior
    # Provider Effects
    provider_effects = posterior['mu_provider']
    provider_means = provider_effects.mean(dim=["chain", "draw"])
    provider_hdi = az.hdi(provider_effects, hdi_prob=hdi_prob)

    provider_dim = list(provider_means.dims)[0]
    provider_names = provider_encoder.inverse_transform(np.arange(provider_means.shape[0]))

    plt.figure(figsize=(10, len(provider_names) * 0.3 + 3))
    for i, name in enumerate(provider_names):
        mean_val = provider_means.sel({provider_dim: i}).values.item()
        lower = provider_hdi.sel({provider_dim: i, "hdi": "lower"})["mu_provider"]
        upper = provider_hdi.sel({provider_dim: i, "hdi": "higher"})["mu_provider"]
        plt.errorbar(mean_val, i,
                     xerr=[[mean_val - lower], [upper - mean_val]],
                     fmt='o', color='black', capsize=4)

    plt.axvline(0, linestyle='--', color='red', alpha=0.5)
    plt.yticks(range(len(provider_names)), provider_names)
    plt.title("Provider Effects (Posterior Mean ± 94% HDI)")
    plt.xlabel("Effect Size")
    plt.tight_layout()
    plt.show()
    
    # Covariate Effects
    beta_effects = posterior['betas']
    beta_means = beta_effects.mean(dim=["chain", "draw"])
    beta_hdi = az.hdi(beta_effects, hdi_prob=hdi_prob)

    beta_dim = list(beta_means.dims)[0]

    plt.figure(figsize=(8, len(covariates) * 0.6 + 2))
    for i, cov in enumerate(covariates):
        mean_val = beta_means.sel({beta_dim: i}).values.item()
        lower = beta_hdi.sel({beta_dim: i, "hdi": "lower"})["betas"]
        upper = beta_hdi.sel({beta_dim: i, "hdi": "higher"})["betas"]
        plt.errorbar(mean_val, i,
                     xerr=[[mean_val - lower], [upper - mean_val]],
                     fmt='o', color='blue', capsize=4)

    plt.axvline(0, linestyle='--', color='red', alpha=0.5)
    plt.yticks(range(len(covariates)), covariates)
    plt.title("Covariate Effects (Posterior Mean ± 94% HDI)")
    plt.xlabel("Effect Size")
    plt.tight_layout()
    plt.show()
