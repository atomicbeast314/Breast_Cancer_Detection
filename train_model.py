#importing the essential libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

#loading the dataset
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

# Visualization disabled: pairplot of all features
# plt.figure(figsize=(30, 30))
# sns.pairplot(
#     df_cancer,
#     vars=cancer['feature_names'],
#     hue='target',
#     diag_kind='kde',
#     plot_kws={'alpha': 0.6, 's': 20}
# )
# plt.suptitle('Pair Plot of All Breast Cancer Features', y=1.02, fontsize=20)
# plt.tight_layout()
# plt.show()

# Split data into features and target
X = df_cancer.drop('target', axis=1)
y = df_cancer['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize different models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name} Results:")
    print("=" * 50)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Update best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

    # Visualization disabled: ROC curve plotting
    # if hasattr(model, "predict_proba"):
    #     y_prob = model.predict_proba(X_test)[:, 1]
    #     fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    #     roc_auc = auc(fpr, tpr)
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'ROC Curve - {name}')
    #     plt.legend(loc="lower right")
    #     plt.show()

# Save the best model
if best_model is not None:
    print(f"\nSaving best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    dump(best_model, 'best_breast_cancer_model.joblib')
    print("Model saved successfully as 'best_breast_cancer_model.joblib'")

# Hyperparameter tuning for Random Forest
print("\nHyperparameter Tuning for Random Forest:")
print("=" * 50)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# Visualization disabled: feature importance barplot
# importances = best_rf.feature_importances_
# feature_importance = pd.DataFrame({
#     'Feature': cancer['feature_names'],
#     'Importance': importances
# }).sort_values('Importance', ascending=False)
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance)
# plt.title('Feature Importance from Random Forest')
# plt.tight_layout()
# plt.show()

# Save the scaler as well (IMPORTANT!)
dump(scaler, 'scaler.joblib')
print("Scaler saved as 'scaler.joblib'")