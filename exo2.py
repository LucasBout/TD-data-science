import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

emplacements_clients = np.array([
    [1, 2], [2, 3], [1, 4], [10, 10], [11, 12], [9, 11], [4, 5], [3, 4], [4, 6]

])
kmeans = KMeans(n_clusters=3, random_state=0).fit(emplacements_clients)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(emplacements_clients[:, 0], emplacements_clients[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.xlabel('Coordonnée X')
plt.ylabel('Coordonnée Y')
plt.title('Clustering des emplacements des clients')
plt.legend()
plt.show()

for i, point in enumerate(emplacements_clients):
    print(f'Client {point} appartient au cluster {labels[i]}')

print("Un point très éloigné des autres clusters pourrait indiquer un client isolé ou une opportunité de marché dans une nouvelle zone géographique.")