import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, accuracy_score,
                           precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('fraud_detection_dataset.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

X = df.drop(['is_fraud', 'timestamp', 'user_id'], axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

categorical_features = ['location', 'device_type']
numeric_features = ['amount', 'age', 'income', 'debt', 'credit_score', 'hour', 'day_of_week', 'day_of_month', 'month']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print('Метрики:')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nПодробный отчет о классификации:")
print(classification_report(y_test, y_pred, target_names=['Легальная', 'Мошенническая']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Легальная', 'Мошенническая'],
            yticklabels=['Легальная', 'Мошенническая'])
plt.xlabel('Предсказанный')
plt.ylabel('Фактический')
plt.title('Матрица ошибок')
plt.show()

joblib.dump(model, 'fraud_detection_model.pkl')
print("Модель сохранена в 'fraud_detection_model.pkl'")
