from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data_path = 'iris.csv'  # Înlocuiește cu calea ta
iris = pd.read_csv(data_path)

# Definim caracteristicile și etichetele
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Evaluare pentru fiecare caracteristică
results = {}
features = X.columns

for feature in features:
    # Selectăm doar caracteristica curentă
    X_feature = X[[feature]].values

    # Aplicăm K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_feature)

    # Calculăm ARI și Silhouette Score
    ari = adjusted_rand_score(y_encoded, clusters)
    silhouette = silhouette_score(X_feature, clusters)

    results[feature] = {'ARI': ari, 'Silhouette': silhouette}

# Afișăm rezultatele
best_feature = max(results, key=lambda x: results[x]['ARI'])
for feature, scores in results.items():
    print(f"Pentru atributul: {feature}")
    print(f"  Adjusted Rand Index (ARI): {scores['ARI']:.4f}")
    print(f"  Silhouette Scor: {scores['Silhouette']:.4f}")

print(f"\nCaracteristica cu cel mai bun ARI: {best_feature} (ARI = {results[best_feature]['ARI']:.4f})")
'''
b) Mai departe se va lua caracteristica cu cel mai bun ARI pentru ca atunci
stim ca aceasta are cea mai mare acuratete la clasificare (valorile de y atribuite sunt
cele similare fata de setul de date initial). In cazul nostru, pentru ca se obtine outputul>

Pentru atributul: sepal_length
  Adjusted Rand Index (ARI): 0.3687
  Silhouette Scor: 0.5613
Pentru atributul: sepal_width
  Adjusted Rand Index (ARI): 0.1569
  Silhouette Scor: 0.5642
Pentru atributul: petal_length
  Adjusted Rand Index (ARI): 0.8680
  Silhouette Scor: 0.6752
Pentru atributul: petal_width
  Adjusted Rand Index (ARI): 0.8857
  Silhouette Scor: 0.7268

Este clar ca caracteristica cu cel mai bun ARI: petal_width (ARI = 0.8857)
'''