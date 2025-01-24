import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

heures_etude = np.array([[1], [2], [3], [4], [5]])
scores = np.array([50, 55, 60, 65, 70])

model = LinearRegression()
model.fit(heures_etude, scores)

heures_nouvel_etudiant = np.array([[6]])
score_prevu = model.predict(heures_nouvel_etudiant)
print(f'Score prévu pour un étudiant ayant étudié 6 heures: {score_prevu[0]}')

correlation, _ = pearsonr(heures_etude.flatten(), scores)
print(f'Corrélation entre le nombre d’heures d’étude et le score obtenu: {correlation}')

plt.scatter(heures_etude, scores, color='blue', label='Données réelles')
plt.plot(heures_etude, model.predict(heures_etude), color='red', label='Modèle de régression')
plt.xlabel('Heures d\'étude')
plt.ylabel('Score')
plt.legend()
plt.show()