# 1. Importación de librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Cargar dataset
df = pd.read_csv("/workspaces/Taller-clasificacion/diabetes.csv")  # Cambia la ruta al CSV descargado

# 3. Exploración de datos (EDA)
print("3. Exploración")
print(df.head())
print(df.info())
print(df.describe())

# 4. Preprocesamiento
# Aquí conviertes variables categóricas, eliminas nulos, normalizas si es necesario
print("4. preprocsamiento")
df = df.dropna()  # Ejemplo básico

# 5. Dividir en variables predictoras (X) y variable objetivo (y)
# X = df.drop("target_column", axis=1)  # Reemplaza con el nombre correcto
# y = df["target_column"]
print("5. Dividir variables predictorias")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# 6. Dividir en entrenamiento y prueba
print("6. Entrenamiento y prueba")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entrenar modelo
print("7. Entrenar modeo")
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 8. Predicciones
print("8. predicciones")
y_pred = model.predict(X_test)

# 9. Evaluación
print("9. Evaluación")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 10. Visualización del árbol
print("10. arbol")
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
plt.show()
