
# Pasul b: Model Bayesian cu PyMC
with pm.Model() as model:
    # Priori pentru parametri
    w = pm.Normal("w", mu=0, sigma=10)
    b = pm.Normal("b", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)


    mu = w * x + b
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(1000, return_inferencedata=True, random_seed=42, progressbar=True)

az.plot_trace(trace)
plt.show()

# Rezumatul statisticilor a posteriori
summary = az.summary(trace, hdi_prob=0.95)
print(summary)

# Comparare parametrii reali vs estimați
w_mean = trace.posterior['w'].mean().item()
b_mean = trace.posterior['b'].mean().item()
sigma_mean = trace.posterior['sigma'].mean().item()

print(f"Valori reale: w={w_real}, b={b_real}, sigma={sigma_real}")
print(f"Valori estimate: w={w_mean:.2f}, b={b_mean:.2f}, sigma={sigma_mean:.2f}")
