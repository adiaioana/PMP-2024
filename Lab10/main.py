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

np.random.seed(42)

w_real = 2.5
b_real = 30
sigma_real = 5

x = np.random.uniform(0, 20, 100)  # Ani de experiență între 0 și 20
epsilon = np.random.normal(0, sigma_real, size=x.shape)
print(epsilon)
exit()
y = w_real * x + b_real + epsilon

# Vizualizare date generate
plt.scatter(x, y, label="Observații generate", alpha=0.7)
plt.xlabel("Ani de experiență")
plt.ylabel("Venituri lunare")
plt.title("Date simulate: Venituri vs Experiență")
plt.legend()
plt.show()
