# Analiză Bayesiană folosind PyMC

## Enunțul Problemei

Un magazin este vizitat de `n` clienți într-o anumită zi. Numărul `Y` de clienți care cumpără un anumit produs este distribuit binomial:
\[ Y \sim \text{Binomial}(n, \theta) \]
unde:
- \( \theta \) este probabilitatea ca un client să cumpere acel produs.

Presupunem că distribuția a priori pentru \( n \) este:
\[ n \sim \text{Poisson}(10) \]

Obiective:
1. Calculați distribuția a posteriori pentru \( n \) folosind PyMC pentru toate combinațiile de:
   - \( Y \in \{0, 5, 10\} \)
   - \( \theta \in \{0.2, 0.5\} \)
2. Vizualizați distribuțiile a posteriori folosind `az.plot_posterior`.
3. Explicați efectul lui \( Y \) și \( \theta \) asupra distribuției a posteriori.

---

## Soluția

### Pasul 1: Modelarea în PyMC
Am folosit PyMC pentru a defini:
- **Distribuția prior pentru \( n \):**
  \[ n \sim \text{Poisson}(10) \]
- **Distribuția pentru \( Y \):**
  \[ Y \sim \text{Binomial}(n, \theta) \]

### Pasul 2: Calcularea Posteriorului
Pentru fiecare combinație de \( Y \) și \( \theta \), am calculat distribuția a posteriori a lui \( n \) folosind metoda MCMC (`pm.sample`).

### Pasul 3: Vizualizare
Distribuțiile a posteriori au fost vizualizate folosind `az.plot_posterior` pentru a evidenția intervalele de încredere.

---

## Rezultate și Observații

1. **Efectul lui \( Y \):**
   - Dacă \( Y \) este mai mare, distribuția a posteriori pentru \( n \) tinde să crească, sugerând că un număr mai mare de clienți a vizitat magazinul.
2. **Efectul lui \( \theta \):**
   - O valoare mai mare a lui \( \theta \) reduce estimarea pentru \( n \), deoarece este nevoie de mai puțini clienți pentru a explica un \( Y \) mare.

---
