# ============================================================
# MILESTONE 4: MODEL OPTIMIZATION
# Student Performance Dataset
# Focus: Reducing Overfitting
# ============================================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import arff

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# -------------------------------
# 2. LOAD DATASET
# -------------------------------
file_path = "data/CEE_DATA.arff"

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

print("Dataset successfully loaded!")


# -------------------------------
# 3. DATA CLEANING
# -------------------------------
# Convert byte values into normal string values
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# Remove duplicate rows
df = df.drop_duplicates()

print("Dataset shape after cleaning:", df.shape)


# -------------------------------
# 4. FEATURE AND TARGET SPLIT
# -------------------------------
X = df.drop("Performance", axis=1)
y = df["Performance"]

print("\nTarget class distribution:")
print(y.value_counts())


# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 6. PREPROCESSING
# -------------------------------
# All features are categorical, so OneHotEncoder is suitable.
categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ============================================================
# PART A: BASELINE MODEL BEFORE OPTIMIZATION
# ============================================================

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)

baseline_model.fit(X_train, y_train)

baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)

baseline_train_acc = accuracy_score(y_train, baseline_train_pred)
baseline_test_acc = accuracy_score(y_test, baseline_test_pred)

print("\n--- BEFORE OPTIMIZATION: BASELINE DECISION TREE ---")
print(f"Training Accuracy: {baseline_train_acc:.4f}")
print(f"Testing Accuracy: {baseline_test_acc:.4f}")

print("\nClassification Report Before Optimization:")
print(classification_report(y_test, baseline_test_pred))


# ============================================================
# PART B: OPTIMIZED MODEL USING GRIDSEARCHCV
# ============================================================

optimized_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)

param_grid = {
    "classifier__max_depth": [2, 3, 4, 5, 6, 8],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 6],
    "classifier__criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    optimized_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

optimized_train_pred = best_model.predict(X_train)
optimized_test_pred = best_model.predict(X_test)

optimized_train_acc = accuracy_score(y_train, optimized_train_pred)
optimized_test_acc = accuracy_score(y_test, optimized_test_pred)

print("\n--- AFTER OPTIMIZATION: TUNED DECISION TREE ---")
print("Best Parameters:", grid_search.best_params_)
print(f"Training Accuracy: {optimized_train_acc:.4f}")
print(f"Testing Accuracy: {optimized_test_acc:.4f}")

print("\nClassification Report After Optimization:")
print(classification_report(y_test, optimized_test_pred))


# ============================================================
# PART C: COMPARE BEFORE AND AFTER OPTIMIZATION
# ============================================================

results = pd.DataFrame({
    "Model": ["Before Optimization", "After Optimization"],
    "Training Accuracy": [baseline_train_acc, optimized_train_acc],
    "Testing Accuracy": [baseline_test_acc, optimized_test_acc],
    "Overfitting Gap": [
        baseline_train_acc - baseline_test_acc,
        optimized_train_acc - optimized_test_acc
    ]
})

print("\n--- MODEL COMPARISON ---")
print(results)


# -------------------------------
# 7. SAVE RESULTS AND VISUALIZATIONS
# -------------------------------
os.makedirs("results", exist_ok=True)

# Accuracy comparison chart
plt.figure(figsize=(8, 5))

x = range(len(results["Model"]))
plt.bar(x, results["Training Accuracy"], width=0.4, label="Training Accuracy")
plt.bar([i + 0.4 for i in x], results["Testing Accuracy"], width=0.4, label="Testing Accuracy")

plt.xticks([i + 0.2 for i in x], results["Model"])
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy Before and After Optimization")
plt.legend()
plt.tight_layout()
plt.savefig("results/milestone4_accuracy_comparison.png")
plt.show()


# Overfitting gap chart
plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["Overfitting Gap"])
plt.ylabel("Training Accuracy - Testing Accuracy")
plt.title("Overfitting Gap Before and After Optimization")
plt.tight_layout()
plt.savefig("results/milestone4_overfitting_gap.png")
plt.show()


# Confusion matrix for optimized model
plt.figure(figsize=(7, 5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix After Optimization")
plt.tight_layout()
plt.savefig("results/milestone4_confusion_matrix.png")
plt.show()


# Save comparison results as CSV
results.to_csv("results/milestone4_model_comparison.csv", index=False)

print("\nMilestone 4 completed successfully.")
print("Charts and comparison results saved in the results folder.")
