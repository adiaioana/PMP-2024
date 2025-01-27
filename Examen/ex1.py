import pandas as pd
import statsmodels.api as sm
#motivez ca as fi vrut sa testez c si d, dar nu merge pymc3 iar a,b le-am implementat in doua moduri pentru ca nu puteam sa compilez versiunea 1 de mai jos (pentru ca nu merge pymc)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import arviz as az
#import pymc3 as pm


# Creăm dataframe-ul din date
data_path = 'date_alegeri_turul2.csv'
data = pd.read_csv(data_path)
df = pd.DataFrame(data)

# Definim variabila dependentă și cele independente
X = df[["Educatie", "Venit"]]  # Variabilele independente
X = sm.add_constant(X)       # Adăugăm o constantă pentru model
y = df["Vot"]                # Variabila dependentă

# Construim modelul de regresie logistică
model = sm.Logit(y, X)
result = model.fit()

# Rezumatul modelului
print(result.summary())
#b) din plot summary, se obtin rezultate mai bune pentru educatie dpdv a P,

# Standardizarea variabilelor
scaler = StandardScaler()
data[['Venit', 'Educatie']] = scaler.fit_transform(data[['Venit', 'Educatie']])

# Modelul logistic
X = data[['Venit', 'Educatie']]
y = data['Vot']
model = LogisticRegression()
model.fit(X, y)

# Granița de decizie
coef = model.coef_[0]
intercept = model.intercept_
x_values = np.linspace(data['Venit'].min(), data['Venit'].max(), 100)
y_values = -(coef[0] * x_values + intercept) / coef[1]

# Predicții și HDI 94%
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Plotare
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Venit', y='Educatie', hue='Vot', palette='coolwarm', alpha=0.8)
plt.plot(x_values, y_values, color='black', label='Granița de decizie')
plt.fill_between(x_values, y_values - 0.06, y_values + 0.06, color='gray', alpha=0.3, label='HDI 94%')
plt.title(f'Model Logistic: Venit & Educatie (Acuratețe: {accuracy:.2f})')
plt.xlabel('Venit (standardizat)')
plt.ylabel('Educatie (standardizat)')
plt.legend()
plt.show()

# Compararea performanței: WAIC și LOO
with pm.Model() as logistic_model:
    # Definirea variabilelor
    b0 = pm.Normal('b0', mu=0, sigma=10)
    b1 = pm.Normal('b1', mu=0, sigma=10)
    b2 = pm.Normal('b2', mu=0, sigma=10)

    # Definirea probabilităților de succes (logistic regression)
    p = pm.Deterministic('p', 1 / (1 + np.exp(-(b0 + b1 * data['Venit'] + b2 * data['Educatie']))))

    # Observațiile
    y_obs = pm.Bernoulli('y_obs', p=p, observed=data['Vot'])

    # Eșantionarea
    trace = pm.sample(2000, return_inferencedata=True)

# Calculează WAIC și LOO
waic = az.waic(trace)
loo = az.loo(trace)

# Afișare rezultate
print(f'WAIC pentru modelul logistic: {waic.waic.values[0]}')
print(f'LOO pentru modelul logistic: {loo.loo.values[0]}')


'''
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az

# Load the dataset
data_path = 'date_alegeri_turul2.csv'
data = pd.read_csv(data_path)

print(data.head())

# pas 1: curatam datele
#data = data[["Varsta","Educatie","Venit","Vot"]] # sexul nu are sens
data = data.dropna()  # Remove rows with missing values
varsta = data[['Varsta']]
educatie = data[['Educatie']]
venit = data[['Venit']]
sex = data[['Sex']]
vot = data[['Vot']]

# Subiectul 1a: Modelul de regresie logistica
with pm.Model() as model:
    # Priorii pentru coeficienții regresiei
    
    #Varsta si sexul le-am considerat independente fata de rezultat.
    
    #beta_varsta = pm.Normal("beta_varsta", mu=0, sigma=10)
    #beta_sex = pm.Normal("beta_sex", mu=0, sigma=10)
    beta_educatie = pm.Normal("beta_educatie", mu=0, sigma=10)
    beta_venit = pm.Normal("beta_venit", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)

    # Logit-ul (funcția liniară)
    logit_p = (
            intercept
            #+ beta_varsta * varsta
            #+ beta_sex * sex
            + beta_educatie * educatie
            + beta_venit * venit
    )

    # Probabilitatea votului
    p = pm.math.sigmoid(logit_p)

    # Observațiile (variabila dependentă)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=vot)

    # Mostrare din posterior
    trace = pm.sample(2000, tune=1000, cores=2)
    pm.summary(trace).round(2)

    az.plot_posterior(trace, var_names=["beta_educatie", "beta_venit"])
'''