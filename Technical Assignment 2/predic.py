import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ambil data
data = pd.read_csv('ai4i2020.csv')
# cek data kosong
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)
# hapus kalo ada data kosong
data = data.dropna()

# tampilkan distribusi data untuk Tool wear [min]
plt.figure(figsize=(12, 6))
sns.histplot(data['Tool wear [min]'], kde=True)
plt.title('Tool Wear Distribution')
plt.show()

# tampilkan korelasi matrix
plt.figure(figsize=(12, 6))
numerical_data = data.select_dtypes(include=['float', 'int'])
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# pendefinisian variabel x dan y
X = data.drop(columns=['Machine failure', 'UDI', 'Product ID'])
y = data['Machine failure']

# hapus kolom yang bukan angka
non_numeric_cols = X.select_dtypes(exclude=['float', 'int']).columns
print("Non-numeric columns:", non_numeric_cols)
X = X.drop(columns=non_numeric_cols)

# membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Pemilihan model dan parameter SVM
model_svc = SVC(probability=True, random_state=42)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(estimator=model_svc, param_grid=param_grid_svc, cv=5, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)
best_model_svc = grid_search_svc.best_estimator_
print("Best parameters for SVM: ", grid_search_svc.best_params_)

# Evaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, roc_auc

metrics_svc = evaluate_model(best_model_svc, X_test, y_test)

print(f'SVM - Accuracy: {metrics_svc[0]:.4f}, Precision: {metrics_svc[1]:.4f}, Recall: {metrics_svc[2]:.4f}, F1-score: {metrics_svc[3]:.4f}, ROC AUC: {metrics_svc[4]:.4f}')


# Confusion matrix untuk SVM
conf_matrix_svc = confusion_matrix(y_test, best_model_svc.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svc, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# grafik probabilitas kegagalan berdasarkan "Rotational speed [rpm]"
for failure_type in y_test.unique():
    df_failure = data[data['Machine failure'] == failure_type]
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df_failure, x='Rotational speed [rpm]')
    plt.title(f'Probabilitas kegagalan berdasarkan Rotational speed [rpm] ({failure_type})')
    plt.ylabel('Probability Density')
    plt.show()
