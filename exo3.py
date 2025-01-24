import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

X = np.array([
    [150, 7.0, 'rouge'], [160, 8, 'rouge'], [140, 6, 'rouge'],
    [170, 7.5, 'orange'], [180, 8.5, 'orange'], [175, 8, 'orange']
])

label_encoder = LabelEncoder()
X[:, 2] = label_encoder.fit_transform(X[:, 2])
X = X.astype(float)
y = np.array(["Pomme", "Pomme", "Pomme", "Orange", "Orange", "Orange"])

model = LogisticRegression()
model.fit(X, y)

fruit = np.array([[165, 7.8, 'rouge']])
fruit[:, 2] = label_encoder.transform(fruit[:, 2])
fruit = fruit.astype(float)
prediction = model.predict(fruit)
print(f'La catégorie prédite pour un fruit de poids 165 g et de taille 7.8 cm est: {prediction[0]}')


