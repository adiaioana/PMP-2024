from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''I have a grid of cols'''
# Dimensiunea gridului
dimensiune_grid = (10, 10)

# Lista de culori predefinite
culori = [
    "red", "blue", "green", "yellow",
    "purple", "orange", "pink", "cyan",
    "brown", "lime"
]
short_to_col = {
    'r':"red", 'bl':"blue", 'gr':"green", 'yl':"yellow",
    'pu':"purple", 'o':"orange", 'pi':"pink", 'c':"cyan",
    'br':"brown", 'l':"lime"
}

# Citirea gridului
df = pd.read_csv('grid_culori.csv', header=None)
grid_culori =df.values
# Generarea secvenței de culori observate
observatii =  [short_to_col['r'], short_to_col['r'],short_to_col['l'],short_to_col['yl'], short_to_col['bl']]

# Mapare culori -> indecși
culoare_to_idx = {culoare: idx for idx, culoare in enumerate(culori)}
idx_to_culoare = {idx: culoare for culoare, idx in culoare_to_idx.items()}

# Transformăm secvența de observații în indecși
observatii_idx = [culoare_to_idx[c] for c in observatii]

# Definim stările ascunse ca fiind toate pozițiile din grid (100 de stări)
numar_stari = dimensiune_grid[0] * dimensiune_grid[1]
stari_ascunse = [(i, j) for i in range(dimensiune_grid[0]) for j in range(dimensiune_grid[1])]
stare_to_idx = {stare: idx for idx, stare in enumerate(stari_ascunse)}
idx_to_stare = {idx: stare for stare, idx in stare_to_idx.items()}

# Matrice de tranziție
transitions = np.zeros((numar_stari, numar_stari))
for i, j in stari_ascunse:
    vecini = [
        (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)  # sus, jos, stânga, dreapta
    ]
    vecini_valizi = [stare_to_idx[(x, y)] for x, y in vecini if 0 <= x < 10 and 0 <= y < 10] + [stare_to_idx[(i,j)]]

    for l,k in stari_ascunse:
        if stare_to_idx[(l,k)] not in vecini_valizi:
            transitions[stare_to_idx[(i,j)]][stare_to_idx[(l,k)]] = 0
            continue
        if (j==0 and stare_to_idx[(l,k)] is stare_to_idx[(i,j)]-1) or (j>0 and (l,k) is (i,j-1)):
            transitions[stare_to_idx[(i,j)]][stare_to_idx[(l,k)]] = 0.4
            continue
        if not (j==0 and stare_to_idx[(l,k)] is stare_to_idx[(i,j)]-1) or (j>0 and (l,k) is (i,j-1)):
            transitions[stare_to_idx[(i,j)]][stare_to_idx[(l,k)]] = 1/len(vecini_valizi)
        else:
            transitions[stare_to_idx[(i,j)]][stare_to_idx[(l,k)]] = 0.6/(len(vecini_valizi)-1)

print((transitions[11]))
# Matrice de emisie
emissions = np.zeros((numar_stari, len(culori)))
for i, j in stari_ascunse:
    for col in range(len(culori)):
        emissions [i*10+j][col] = 1.0 if grid_culori[i][j] ==idx_to_culoare[col] else 0.0
######

# Vector de probl. initiala pt fiecare stare (poz din matrice)
initial_probabilities = np.array([1/100]*100)

# Modelul HMM
# Secvența de observații
# Configurarea modelului HMM Gaussian
means = np.array([[0], [1], [2], [3],[4],[5],[6],[7],[8],[9]])  #  10 cols.
covariance_matrices = np.tile(np.identity(1), (numar_stari, 1, 1))  # Covarianțe

# Crearea modelului HMM
hmm_model = hmm.GaussianHMM(n_components=numar_stari, covariance_type="full")
hmm_model.startprob_ = initial_probabilities
hmm_model.transmat_ = transitions
hmm_model.means_ = means
hmm_model.covars_ = covariance_matrices

# Aplicarea algoritmului Viterbi
log_prob, secventa_stari = hmm_model.decode(observatii_idx, algorithm="viterbi")
most_probable_states = [idx_to_culoare[state] for state in secventa_stari]
######
# Rulăm algoritmul Viterbi pentru secvența de observații
######

# Convertim secvența de stări în poziții din grid
drum = [idx_to_stare[idx] for idx in secventa_stari]

# Vizualizăm drumul pe grid
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(dimensiune_grid[0]):
    for j in range(dimensiune_grid[1]):
        culoare = grid_culori[i, j]
        ax.add_patch(plt.Rectangle((j, dimensiune_grid[0] - i - 1), 1, 1, color=culoare))
        ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, culoare,
                color="white", ha="center", va="center", fontsize=8, fontweight="bold")

# Evidențiem drumul rezultat
for idx, (i, j) in enumerate(drum):
    ax.add_patch(plt.Circle((j + 0.5, dimensiune_grid[0] - i - 0.5), 0.3, color="black", alpha=0.7))
    ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, str(idx + 1),
            color="white", ha="center", va="center", fontsize=10, fontweight="bold")

# Setări axă
ax.set_xlim(0, dimensiune_grid[1])
ax.set_ylim(0, dimensiune_grid[0])
ax.set_xticks(range(dimensiune_grid[1]))
ax.set_yticks(range(dimensiune_grid[0]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(visible=True, color="black", linewidth=0.5)
ax.set_aspect("equal")
plt.title("Drumul rezultat al stărilor ascunse", fontsize=14)
plt.show()