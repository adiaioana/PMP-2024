'''

Exercițiu laborator 9
Regresie liniară Bayesiană
Avem un set de date in care dorim să prezicem veniturile lunare ale unei persoane în funcție de numărul de ani de experiență profesională. Datele conţin informații despre experienta profesională x şi veniturile observate y. Presupunem că relația dintre acestea este liniară, dar dorim să utilizăm o abordare bayesiană pentru a estima parametrii modelului, luând în considerare incertitudinea asociată.
a) Generaţi 100 de observații, cu numărul de ani de experiență între 0 şi 20 (aleator), iar venitul lunar folosind un model liniar cu zgomot adăugat.
b) Cu ajutorul PyMC, estimați parametrii folosind un model de regresie liniară Bayesiană. Corespund aceştia cu cei aleşi de voi la punctul a)?
'''

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Pasul a: Generare date
np.random.seed(42)

# Parametrii reali
w_real = 2.5  # Coeficientul pantei
b_real = 30  # Interceptul
sigma_real = 5  # Deviația standard a zgomotului

# Generăm datele
x = np.random.uniform(0, 20, 100)  # Ani de experiență între 0 și 20
epsilon = np.random.normal(0, sigma_real, size=x.shape)  # Zgomot aleator
y = w_real * x + b_real + epsilon  # Venituri simulate

# Vizualizare date generate
plt.scatter(x, y, label="Observații generate", alpha=0.7)
plt.xlabel("Ani de experiență")
plt.ylabel("Venituri lunare")
plt.title("Date simulate: Venituri vs Experiență")
plt.legend()
plt.show()

# Pasul b: Model Bayesian cu PyMC
with pm.Model() as model:
    # Priori pentru parametri
    w = pm.Normal("w", mu=0, sigma=10)  # Priori pentru panta
    b = pm.Normal("b", mu=0, sigma=10)  # Priori pentru intercept
    sigma = pm.HalfNormal("sigma", sigma=10)  # Priori pentru deviația standard

    # Modelul liniar
    mu = w * x + b

    # Observații
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Sampling
    trace = pm.sample(1000, return_inferencedata=True, random_seed=42, progressbar=True)

# Vizualizare rezultatele sampling-ului
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
