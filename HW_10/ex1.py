import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

# *** Citirea datelor din fisierul 'date.csv' ***
# Fisierul trebuie să conțină două coloane (x și y) fără antet.
data = pd.read_csv('data/date.csv', header=None)  # Presupunem că datele nu au un antet
initial_data = data[0].values
x=[float(str[:str.find(' ')]) for str in initial_data]
y=[float(str[str.find(' ')+1:]) for str in initial_data]

# *** Functie pentru a ajusta un model polinomial și a vizualiza rezultatele ***
def fit_polynomial(exNumb,order, noise_std=10, x_data=None, y_data=None):
    """Ajustează un model polinomial de ordin dat și reprezintă grafic datele și modelul."""
    plt.figure(figsize=(10, 5))

    # Dacă nu sunt date noi, folosim cele originale
    if x_data is None or y_data is None:
        x_data = x
        y_data = y

    # Adăugăm zgomot normal la date

    y_noisy = y_data + np.random.normal(0, noise_std, len(y_data))
    plt.plot(x_data, y_noisy, 'o', label=f'Date originale cu zgomot (sd={noise_std})')

    # Ajustare model polinomial
    x_n = np.linspace(min(x_data), max(x_data), 100)
    coeffs = np.polyfit(x_data, y_noisy, deg=order)
    ffit = np.polyval(coeffs, x_n)
    plt.plot(x_n, ffit, label=f'Model polinomial (ordine {order}, sd={noise_std})')

    # Calculul coeficientului R^2
    yhat = np.polyval(coeffs, x_data)
    ybar = np.mean(y_noisy)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y_noisy - ybar) ** 2)
    r2 = ssreg / sstot

    plt.title(f'Ordine {order}, R^2 = {r2:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'plots/{exNumb}/ord{order}_poly_sd{noise_std}_r2({r2:.2f}).png')
    plt.show()
    return coeffs, r2


# *** Partea 1.1a: Ajustare model cu ordine 5 și zgomot sd=10 ***
fit_polynomial(1,order=5, noise_std=10)

# *** Partea 1.1b: Ajustare model cu ordine 5 și zgomot sd=100 ***
fit_polynomial(1,order=5, noise_std=100)


# *** Partea 1.1b: Ajustare cu zgomot personalizat (vector sd) ***
def fit_polynomial_with_custom_sd(exNumb, order, noise_std_array, x_data=None, y_data=None):
    """Ajustează un model polinomial cu zgomot personalizat pentru fiecare punct."""
    plt.figure(figsize=(10, 5))

    # Dacă nu sunt date noi, folosim cele originale
    if x_data is None or y_data is None:
        x_data = x
        y_data = y

    # Generăm zgomot pentru fiecare punct pe baza sd-urilor personalizate
    if len(noise_std_array) != len(y_data):
        noise_std_array = np.resize(noise_std_array, len(y_data))
    noise = np.random.normal(0, noise_std_array, len(y_data))
    y_noisy = y_data + noise
    plt.plot(x_data, y_noisy, 'o', label=f'Date originale cu zgomot personalizat')

    # Ajustare model polinomial
    x_n = np.linspace(min(x_data), max(x_data), 100)
    coeffs = np.polyfit(x_data, y_noisy, deg=order)
    ffit = np.polyval(coeffs, x_n)
    plt.plot(x_n, ffit, label=f'Model polinomial (ordine {order}, zgomot personalizat)')

    # Calculul coeficientului R^2
    yhat = np.polyval(coeffs, x_data)
    ybar = np.mean(y_noisy)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y_noisy - ybar) ** 2)
    r2 = ssreg / sstot

    plt.title(f'Ordine {order}, R^2 = {r2:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'plots/{exNumb}/ord{order}_poly_sd-array_r2({r2:.2f}).png')
    plt.show()
    return coeffs, r2


# Vectorul de zgomot personalizat
custom_sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
fit_polynomial_with_custom_sd(1,order=5, noise_std_array=custom_sd)

# *** Partea 1.2: Creștem numărul de puncte la 500 ***
np.random.seed(42)  # Asigurăm reproducibilitate
x_large = np.linspace(0, 10, 500)
y_large = 2 * x_large ** 2 - 3 * x_large + np.random.normal(0, 10, 500)

# Ajustare pe setul extins de date
fit_polynomial(1,order=5, noise_std=10, x_data=x_large, y_data=y_large)
fit_polynomial(1,order=5, noise_std=100, x_data=x_large, y_data=y_large)
custom_sd_large = np.random.choice([10, 0.1, 0.1, 0.1, 0.1], size=500)
fit_polynomial_with_custom_sd(1,order=5, noise_std_array=custom_sd_large, x_data=x_large, y_data=y_large)


# *** Partea 1.3: Model cubic (order=3), calcul WAIC și LOO ***
def calculate_waic(y_true, y_pred):
    """Calculează WAIC pentru un model dat."""
    likelihood = -0.5 * np.sum((y_true - y_pred) ** 2)
    return likelihood


def calculate_loo(y_true, y_pred):
    """Calculează LOO pentru un model dat."""
    n = len(y_true)
    loo_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
    return np.mean(loo_errors)


# Ajustare model cubic
coeffs_cubic, r2_cubic = fit_polynomial(1,order=3, x_data=x_large, y_data=y_large)

# Predictii pentru WAIC si LOO
x_eval = np.linspace(x_large.min(), x_large.max(), 500)
y_pred_cubic = np.polyval(coeffs_cubic, x_eval)
y_pred_true = np.polyval(coeffs_cubic, x_large)

waic_cubic = calculate_waic(y_large, y_pred_true)
loo_cubic = calculate_loo(y_large, y_pred_true)

print(f"Model cubic:\nR^2 = {r2_cubic:.2f}\nWAIC = {waic_cubic:.2f}\nLOO = {loo_cubic:.2f}")

# Comparație cu modele liniare și pătratice
coeffs_linear, r2_linear = fit_polynomial(1,order=1, x_data=x_large, y_data=y_large)
coeffs_quadratic, r2_quadratic = fit_polynomial(1,order=2, x_data=x_large, y_data=y_large)

# Predictii pentru modelele liniare si pătratice
y_pred_linear = np.polyval(coeffs_linear, x_large)
y_pred_quadratic = np.polyval(coeffs_quadratic, x_large)

waic_linear = calculate_waic(y_large, y_pred_linear)
loo_linear = calculate_loo(y_large, y_pred_linear)
waic_quadratic = calculate_waic(y_large, y_pred_quadratic)
loo_quadratic = calculate_loo(y_large, y_pred_quadratic)

print(f"\nModel liniar:\nR^2 = {r2_linear:.2f}\nWAIC = {waic_linear:.2f}\nLOO = {loo_linear:.2f}")
print(f"\nModel pătratic:\nR^2 = {r2_quadratic:.2f}\nWAIC = {waic_quadratic:.2f}\nLOO = {loo_quadratic:.2f}")

# Reprezentăm grafic comparativ
plt.figure(figsize=(10, 5))
plt.plot(x_large, y_large, 'o', label='Date originale')
plt.plot(x_eval, y_pred_cubic, label=f'Model cubic (R^2 = {r2_cubic:.2f})')
plt.plot(x_large, y_pred_linear, label=f'Model liniar (R^2 = {r2_linear:.2f})')
plt.plot(x_large, y_pred_quadratic, label=f'Model pătratic (R^2 = {r2_quadratic:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparație modele: liniar, pătratic, cubic')

plt.savefig(f'plots/1/Comp.png')
plt.show()

'''
Model cubic:
R^2 = 0.92
WAIC = -24221.90
LOO = 96.89

Model liniar:
R^2 = 0.85
WAIC = -74860.78
LOO = 299.44

Model pătratic:
R^2 = 0.93
WAIC = -24149.03
LOO = 96.60
'''