# Importăm biblioteci necesare
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np

# 1. Încărcăm datele din fișierul CSV
file_path = '/mnt/data/date_promovare_examen-1.csv'  # Actualizează calea fișierului
data = pd.read_csv(file_path)

# Verificăm primele rânduri din date
print("Primele rânduri din date:\n", data.head())

# 2. Verificăm dacă datele sunt echilibrate (distribuția claselor)
print("\nDistribuția claselor (0 = Ne-promovat, 1 = Promovat):")
print(data['promovat'].value_counts())

# Vizualizare distribuție
data['promovat'].value_counts().plot(kind='bar')
plt.title("Distribuția Claselor")
plt.xlabel("Promovat")
plt.ylabel("Număr de Studenți")
plt.show()

# 3. Construim modelul de regresie logistică bayesiană folosind PyMC
# Normalizăm datele de intrare pentru o convergență mai bună
data['ore_studiu_norm'] = (data['ore_studiu'] - data['ore_studiu'].mean()) / data['ore_studiu'].std()
data['ore_somn_norm'] = (data['ore_somn'] - data['ore_somn'].mean()) / data['ore_somn'].std()

# Variabile predictori și răspuns
X_studiu = data['ore_studiu_norm'].values
X_somn = data['ore_somn_norm'].values
y = data['promovat'].values

# Modelul în PyMC
with pm.Model() as logistic_model:
    # Priori pentru coeficienți și intercept
    beta_studiu = pm.Normal("beta_studiu", mu=0, sigma=1)
    beta_somn = pm.Normal("beta_somn", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Likelihood (funcția de verosimilitate)
    p = pm.math.sigmoid(intercept + beta_studiu * X_studiu + beta_somn * X_somn)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    # Sampling pentru inferență bayesiană
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

# 4. Rezultate și analiză
# Rezumatul inferenței
print("\nRezumatul inferenței pentru coeficienți:")
print(az.summary(trace, hdi_prob=0.95))

# Vizualizarea distribuției coeficienților
az.plot_posterior(trace, var_names=["beta_studiu", "beta_somn", "intercept"], ref_val=0)
plt.show()

# Analiză marginală pentru decizia modelului
# Care variabilă influențează mai mult? Comparăm distribuțiile coeficienților.
print("\nConcluzii:")
print("Dacă beta_studiu are valori mai mari decât beta_somn, orele de studiu sunt mai influente.")
print("Dacă beta_somn este dominant, orele de somn sunt mai importante.")
