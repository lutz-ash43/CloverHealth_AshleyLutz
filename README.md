# CloverHealth_AshleyLutz

# Physician Efficacy Evaluation with Bayesian and Empirical Methods
## Overview

This project evaluates the effectiveness of physicians in treating patients with a particular medical condition. The goal is to determine whether any providers have significantly better or worse outcomes than expected after adjusting for patient-level covariates.

Two main approaches are employed:

 - Hierarchical Bayesian Logistic Regression

 - Covariate-Adjusted Empirical Null Testing (with bootstrapping for robustness)

The notebook performs:

 - Exploratory Data Analysis (EDA)

 - Variance Inflation Factor (VIF) analysis for multicollinearity

 - Provider-level outcome evaluation using two robust statistical approaches

 - Visualization of key results to identify physicians whose performance deviates from expectations

## Bottom Line Up Front 
- Through Hierarchical bayesian methods it was determined there was no provider effect when accounting for the covariates of health_risk_assesment and sex (age was excluded due to colienarity with health_risk_assesment). Boostrapped evaluation of individual providers against a covariate adjusted null identfied one provider with increased failure proportion. While this result is significant this provider is such a small percentage of the provider population, and there are several unknown factors such as SDOH, disease duration, location of care, BMI, ect. that at this point I would not recommend excluding any individual physician strategically without more further information. 


## Repo Structure 
```bash
repo/
├── data/
│   └── Product_Data.csv          # Patient and treatment data
├── empirical_null.py             # Empirical null + bootstrap-based analysis
├── hierarchical_bayes.py         # Hierarchical Bayesian model implementation
├── survival_analysis.py          # (Optional) Cox proportional hazards model
├── utils.py                      # Utility functions (VIF, plotting, etc.)
├── analysis_recommendation.ipynb    # Main notebook for analysis and visualization
└── plots                         # plots
│   └── plotly plots from analysis_recommendation.ipynb   
└── README.md                     # This file
```

## Methods 
1. Hierarchical Bayesian Logistic Regression
Implemented in hierarchical_bayes.py using PyMC.

- Models patient outcome (pass/fail) as a function of:
- Covariates: member_sex, health_risk_assesment
- Random effect: servicing_provider_id

- Output includes:
- Posterior distributions for provider-specific effects
- Credible intervals (HDIs) to assess statistical significance

2. Empirical Null with Covariate Adjustment with bootstrap 
- Implemented in empirical_null.py.

For each provider:

- Trains logistic regression on covariates to predict outcomes from other providers
- Uses the resulting model to simulate expected failure distributions

Computes:
- Observed vs expected failure rate
- Confidence intervals via binomial simulations
- p-values to flag significantly poor-performing physicians

Bootstrap 
- repeats this process 100 times to ensure stable results 

## Assumptions 
 - Missing outcome values (NaN) are considered successful outcomes
 - The data represents the full population (no sampling adjustment needed)


## Follow up analysis 
- survival_analysis.py synthesizes time to followup and event times to enable a mock survival analysis using a bayesian cox model to account for covariates. 
- This is meant to provide an example for analysese that would be possible with additional data to enrich and remove assumptions

