import pymc as pm
import arviz as az


values_Y = [5, 10]
values_theta = [0.2, 0.5]
prior_lambda = 10  # Pentru Poisson

posterior_results = []

for Y in values_Y:
    for theta in values_theta:
        with pm.Model() as model:
            # Distribuția prior pentru n
            n = pm.Poisson("n", mu=prior_lambda)

            # Distribuția pentru Y
            Y_obs = pm.Binomial("Y", n=n, p=theta, observed=Y)

            # Mostrăm posteriorul
            trace = pm.sample(1000, return_inferencedata=True)
            posterior_results.append((Y, theta, trace))

for Y, theta, trace in posterior_results:
    print(f"Distribuția a posteriori pentru Y={Y}, theta={theta}")
    az.plot_posterior(trace, var_names=["n"], hdi_prob=0.95)
