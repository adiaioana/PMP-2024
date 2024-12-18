import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

file_path = 'date_promovare_examen.csv'
data = pd.read_csv(file_path)

print("Primele rânduri din date:\n", data.head())

print("\nDistribuția claselor (0 = Ne-promovat, 1 = Promovat):")
print(data['Promovare'].value_counts())

data['Promovare'].value_counts().plot(kind='bar')
plt.title("Distribuția Claselor")
plt.xlabel("Promovat")
plt.ylabel("Număr de Studenți")
plt.show()

data['ore_studiu_norm'] = (data['Ore_Studiu'] - data['Ore_Studiu'].mean()) / data['Ore_Studiu'].std()
data['ore_somn_norm'] = (data['Ore_Somn'] - data['Ore_Somn'].mean()) / data['Ore_Somn'].std()

X_studiu = data['ore_studiu_norm'].values
X_somn = data['ore_somn_norm'].values
y = data['Promovare'].values

with pm.Model() as logistic_model:
    beta_studiu = pm.Normal("beta_studiu", mu=0, sigma=1)
    beta_somn = pm.Normal("beta_somn", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    p = pm.math.sigmoid(intercept + beta_studiu * X_studiu + beta_somn * X_somn)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

print("\nRezumatul inferenței pentru coeficienți:")
print(az.summary(trace, hdi_prob=0.95))

az.plot_posterior(trace, var_names=["beta_studiu", "beta_somn", "intercept"], ref_val=0)
plt.show()

print("\nConcluzii:")
print("Dacă beta_studiu are valori mai mari decât beta_somn, orele de studiu sunt mai influente.")
print("Dacă beta_somn este dominant, orele de somn sunt mai importante.")
