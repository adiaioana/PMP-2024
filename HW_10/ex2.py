from ex1 import fit_polynomial, calculate_waic, calculate_loo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('.\data\date_studiu_nota.csv')  # Presupunem că fișierul are un antet

x = data['Ore_Studiu'].values
y = data['Nota_Finala'].values

# *** Partea 2.1: Modelarea relației dintre orele de studiu și nota finală ***
# Model liniar
coeffs_linear, r2_linear = fit_polynomial(2,order=1, x_data=x, y_data=y)

# Model pătratic
coeffs_quadratic, r2_quadratic = fit_polynomial(2,order=2, x_data=x, y_data=y)

# Model cubic
coeffs_cubic, r2_cubic = fit_polynomial(2,order=3, x_data=x, y_data=y)

# *** Partea 2.2: Determinarea celui mai bun model folosind WAIC și LOO ***
# Predictii pentru fiecare model
y_pred_linear = np.polyval(coeffs_linear, x)
y_pred_quadratic = np.polyval(coeffs_quadratic, x)
y_pred_cubic = np.polyval(coeffs_cubic, x)

# Calculul WAIC și LOO pentru fiecare model
waic_linear = calculate_waic(y, y_pred_linear)
loo_linear = calculate_loo(y, y_pred_linear)

waic_quadratic = calculate_waic(y, y_pred_quadratic)
loo_quadratic = calculate_loo(y, y_pred_quadratic)

waic_cubic = calculate_waic(y, y_pred_cubic)
loo_cubic = calculate_loo(y, y_pred_cubic)

# Afișăm rezultatele
print(f"\nModel liniar:\nR^2 = {r2_linear:.2f}\nWAIC = {waic_linear:.2f}\nLOO = {loo_linear:.2f}")
print(f"\nModel pătratic:\nR^2 = {r2_quadratic:.2f}\nWAIC = {waic_quadratic:.2f}\nLOO = {loo_quadratic:.2f}")
print(f"\nModel cubic:\nR^2 = {r2_cubic:.2f}\nWAIC = {waic_cubic:.2f}\nLOO = {loo_cubic:.2f}")

# *** Reprezentarea grafică a modelelor ***
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Date originale')

# Generăm valori pentru curbele modelate
x_eval = np.linspace(x.min(), x.max(), 500)
y_eval_linear = np.polyval(coeffs_linear, x_eval)
y_eval_quadratic = np.polyval(coeffs_quadratic, x_eval)
y_eval_cubic = np.polyval(coeffs_cubic, x_eval)

plt.plot(x_eval, y_eval_linear, label=f'Model liniar (R^2 = {r2_linear:.2f})')
plt.plot(x_eval, y_eval_quadratic, label=f'Model pătratic (R^2 = {r2_quadratic:.2f})')
plt.plot(x_eval, y_eval_cubic, label=f'Model cubic (R^2 = {r2_cubic:.2f})')

plt.xlabel('Ore de studiu')
plt.ylabel('Nota finală')
plt.title('Comparație modele: liniar, pătratic, cubic')
plt.legend()
plt.savefig(f'plots/2/Comp.png')
plt.show()

'''
Model liniar:
R^2 = 0.04
WAIC = -153.00
LOO = 3.06

Model pătratic:
R^2 = 0.02
WAIC = -59.06
LOO = 1.18

Model cubic:
R^2 = 0.03
WAIC = -190.76
LOO = 3.82
'''