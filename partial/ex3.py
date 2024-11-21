import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def make_experiment_post(experiment):
    no_s = len([el for el in experiment if el == 's'])
    no_t = len(experiment)
    observed_rate = no_s / no_t  # Rata observată = 7/10=0.7

    # Parametrii priori pentru distribuția Gamma (neinformativ, pentru ilustrare)
    alpha_prior = 1  # Parametrul de formă pentru prior
    beta_prior = 1  # Parametrul de rată pentru prior

    # Parametrii posteriori
    alpha_posterior = alpha_prior + no_s  # Actualizează alpha
    beta_posterior = beta_prior + no_t  # Actualizează beta

    # Eșantionare din distribuția posterior Gamma
    posterior_samples = np.random.gamma(alpha_posterior, 1 / beta_posterior, 10000)

    # Folosim arviz pentru a plota distribuția posterior și intervalul HDI de 94%
    az.plot_posterior(posterior_samples, hdi_prob=0.94)
    plt.title("Distribuția Posterior a Rată Steme (λ)")
    plt.xlabel("Rata Steme λ (obtinute in cele 10 aruncari)")
    plt.show()

    # Afișăm intervalul HDI de 94%
    hdi_bounds = az.hdi(posterior_samples, hdi_prob=0.94)
    print(f"94% HDI pentru λ: [{hdi_bounds[0]:.2f}, {hdi_bounds[1]:.2f}]")

def make_experiment_prior(experiment):
    no_s = len([el for el in experiment if el == 's'])
    no_t = len(experiment)
    observed_rate = no_s / no_t  # Rata observată = 7/10=0.7

    # Parametrii priori pentru distribuția Gamma (neinformativ, pentru ilustrare)
    alpha_prior = 1  # Parametrul de formă pentru prior
    beta_prior = 1  # Parametrul de rată pentru prior

    # Parametrii posteriori


    # Eșantionare din distribuția prior Gamma
    prior_samples = np.random.gamma(alpha_prior, 1 / beta_prior, 10000)

    # Folosim arviz pentru a plota distribuția posterior și intervalul HDI de 94%
    az.plot_posterior(prior_samples, hdi_prob=0.94)
    plt.title("Distribuția Prior a Rată Steme (λ)")
    plt.xlabel("Rata Steme λ (obtinute in cele 10 aruncari)")
    plt.show()

    # Afișăm intervalul HDI de 94%
    hdi_bounds = az.hdi(prior_samples, hdi_prob=0.94)
    print(f"94% HDI pentru λ: [{hdi_bounds[0]:.2f}, {hdi_bounds[1]:.2f}]")


experiment_1 = ['s','s','b','b','s','s','s','s','b','s']
make_experiment_prior(experiment_1)

experiment_2 = ['b','b','s','b','s','s','b','s','s','b']
make_experiment_post(experiment_2)
