import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# 1. Cargar datos
df_values = pd.read_csv(r"C:\Users\isard\Downloads\Terremotos\train_values.csv")
df_labels = pd.read_csv(r"C:\Users\isard\Downloads\Terremotos\train_labels.csv")
df_test = pd.read_csv(r"C:\Users\isard\Downloads\Terremotos\test_values.csv")

# 1.1 Función para transformar columnas object a numéricas
def transform_object_columns(df_orig):
    df_out = df_orig.copy()
    
    obj_cols = df_out.select_dtypes(include="object").columns
    for col in obj_cols:
        # Valores únicos en la columna
        unique_vals = set(df_out[col].dropna().unique())
        
        # Si solo contiene 't' y/o 'f', mapeamos a 1 y 0
        if unique_vals <= {'t', 'f'}:
            df_out[col] = df_out[col].map({'t': 1, 'f': 0})
        else:
            # De lo contrario, aplicamos LabelEncoder
            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col].astype(str))
    
    return df_out

# 1.2 Transformar 't'/'f' (y otras strings) en df_values y df_test
df_values_transformed = transform_object_columns(df_values)
df_test_transformed = transform_object_columns(df_test)

# 1.3 Fusionar datasets de entrenamiento
df = df_values_transformed.merge(df_labels, on="building_id")

# 1.4 Separar features y target
X = df.drop(columns=["building_id", "damage_grade"])
y = df["damage_grade"]

# 2. Usar un subconjunto para pruebas (para reducir tiempo)
X_sample, _, y_sample, _ = train_test_split(
    X, y, stratify=y, train_size=0.3, random_state=42
)

# 3. Dendograma para selección de características
X_num = X_sample.select_dtypes(include='number')
X_num_sample = X_num.sample(n=500, random_state=42)  # Muestra aleatoria para no consumir mucha memoria
linked = linkage(X_num_sample, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=20)
plt.title("Dendograma para selección de características")
plt.show()

# 4. LazyPredict para análisis inicial (OPCIONAL, comentar si tarda mucho)
# lazy = LazyClassifier(verbose=0, ignore_warnings=True)
# models_lazy, _ = lazy.fit(X_sample, X_sample, y_sample, y_sample)
# print(models_lazy)

# 5. Entrenamiento de modelos
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42, verbose=1)
}

print("\n🚀 Iniciando entrenamiento de modelos...")
for name, model in models.items():
    print(f"\n📊 Entrenando {name}...")
    model.fit(X_sample, y_sample)
    y_pred = model.predict(X_sample)
    score = f1_score(y_sample, y_pred, average="micro")
    print(f"✅ {name} Micro F1 Score: {score:.4f}")

# 6. Optimización de hiperparámetros con GridSearchCV y RandomizedSearchCV
print("\n🔍 Iniciando optimización de hiperparámetros...")

# 6.1 GridSearchCV para Random Forest
rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20, None]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=3,
    scoring='f1_micro',
    verbose=2
)
print("\n📊 Optimizando Random Forest...")
grid_rf.fit(X_sample, y_sample)
print(f"✅ Mejor RandomForest: {grid_rf.best_params_}")
print(f"✅ Mejor score: {grid_rf.best_score_:.4f}")

# 6.2 RandomizedSearchCV para SVM
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf"]
}
random_svm = RandomizedSearchCV(
    SVC(random_state=42),
    svm_params,
    n_iter=4,
    cv=3,
    scoring='f1_micro',
    verbose=2
)
print("\n📊 Optimizando SVM...")
random_svm.fit(X_sample, y_sample)
print(f"✅ Mejor SVM: {random_svm.best_params_}")
print(f"✅ Mejor score: {random_svm.best_score_:.4f}")

# 7. Entrenar el modelo final con TODOS los datos y crear CSV de submission
print("\n🎯 Entrenando modelo final...")
X_full = df.drop(columns=["building_id", "damage_grade"])
y_full = df["damage_grade"]

final_model = grid_rf.best_estimator_
final_model.fit(X_full, y_full)

print("\n📝 Generando predicciones para el conjunto de test...")

# Asegurarnos de usar el df_test_transformado (ya convertido a numérico)
X_test = df_test_transformed.drop(columns="building_id")
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    "building_id": df_test_transformed["building_id"],
    "damage_grade": test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Archivo submission.csv generado exitosamente")
