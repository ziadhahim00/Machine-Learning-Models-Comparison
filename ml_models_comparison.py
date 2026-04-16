# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Regression models
from sklearn.linear_model import Ridge, Lasso

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Load datasets
insurance = pd.read_csv("insurance.csv")
student = pd.read_csv("student.csv", )
heart = pd.read_csv("heart.csv")
diabetes = pd.read_csv("diabetes.csv")
mushroom = pd.read_csv("mushroom.csv" , sep=';')
wine = pd.read_csv("Wine.csv", )

# =========================
# Regression Part
# =========================

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Different lambda values
lambdas = [0.001, 0.01, 0.1, 1, 10, 100]

def run_regression_experiment(df, target_col, dataset_name):
    print(f"\n{'='*50}")
    print(f"Regression on {dataset_name}")
    print(f"{'='*50}")

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert categorical columns to numbers
    X = pd.get_dummies(X, drop_first=True)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for lam in lambdas:
        model = Ridge(alpha=lam)

        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=kf, scoring='r2'
        )
        mean_cv_score = np.mean(cv_scores)

        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Lambda": lam,
            "CV_R2_Mean": round(mean_cv_score, 4),
            "Test_MSE": round(mse, 4),
            "Test_RMSE": round(rmse, 4),
            "Test_MAE": round(mae, 4),
            "Test_R2": round(r2, 4)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Best lambda
    best_row = results_df.loc[results_df["CV_R2_Mean"].idxmax()]

    print("\nResults Table:")
    print(results_df)

    print(f"\nBest lambda for {dataset_name}: {best_row['Lambda']}")
    print(f"Best CV R2: {best_row['CV_R2_Mean']}")

    return results_df
# Run regression on Insurance
insurance_results = run_regression_experiment(
    insurance,
    target_col="charges",
    dataset_name="Insurance"
)

# Run regression on Student
student_results = run_regression_experiment(
    student,
    target_col="G3",
    dataset_name="Student Performance"
)

# =========================
# Classification Part
# =========================

def run_classification_experiment(df, target_col, dataset_name):
    print(f"\n{'='*50}")
    print(f"Classification on {dataset_name}")
    print(f"{'='*50}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert categorical to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("\nConfusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test, y_pred_lr))

    results.append({
        "Model": "Logistic Regression",
        "Accuracy": round(accuracy_score(y_test, y_pred_lr), 4),
        "Precision": round(precision_score(y_test, y_pred_lr, average='weighted'), 4),
        "Recall": round(recall_score(y_test, y_pred_lr, average='weighted'), 4),
        "F1": round(f1_score(y_test, y_pred_lr, average='weighted'), 4)
    })

    # Neural Network
    nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)

    print("\nConfusion Matrix (Neural Network):")
    print(confusion_matrix(y_test, y_pred_nn))

    results.append({
        "Model": "Neural Network",
        "Accuracy": round(accuracy_score(y_test, y_pred_nn), 4),
        "Precision": round(precision_score(y_test, y_pred_nn, average='weighted'), 4),
        "Recall": round(recall_score(y_test, y_pred_nn, average='weighted'), 4),
        "F1": round(f1_score(y_test, y_pred_nn, average='weighted'), 4)
    })

    results_df = pd.DataFrame(results)

    print("\nResults Table:")
    print(results_df)

    return results_df
# Heart
run_classification_experiment(heart, "target", "Heart")

# Diabetes
run_classification_experiment(diabetes, "Outcome", "Diabetes")

# Mushroom
run_classification_experiment(mushroom, "class", "Mushroom")

# Wine (حولها إلى Classification)
wine["quality_class"] = wine["quality"].apply(lambda x: 1 if x >= 6 else 0)

run_classification_experiment(
    wine.drop(columns=["quality"]),
    "quality_class",
    "Wine"
)




# Data
datasets = ['Heart', 'Diabetes', 'Mushroom', 'Wine']
lr_accuracy = [0.7951, 0.7727, 0.8392, 0.7686]
nn_accuracy = [0.9854, 0.8377, 1.0000, 0.7555]

# Positions
x = np.arange(len(datasets))
width = 0.35

# Plot
plt.figure(figsize=(10,6))
plt.bar(x - width/2, lr_accuracy, width, label='Logistic Regression')
plt.bar(x + width/2, nn_accuracy, width, label='Neural Network')

# Labels and title
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Logistic Regression vs Neural Network')
plt.xticks(x, datasets)
plt.ylim(0.7, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()