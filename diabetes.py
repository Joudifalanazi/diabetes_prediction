#import important libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load our dataset
def load_dataset():
    print("Loading dataset...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, names=column_names)
    return data

#   dataset info
def explore_dataset(data):
    print("Dataset Info:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())

# Preprocess the data
def preprocess_data(data):
    # Replace 0s with NaN for columns where 0 doesn't make sense
    data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)

    # Fill missing values with the mean
    data.fillna(data.mean(), inplace=True)

    # Split features and target
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train and test models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Logistic Regression
    print("\nTraining Logistic Regression model...")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    print("Logistic Regression Results:")
    evaluate_model(log_reg, X_test, y_test)

    # Random Forest
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("Random Forest Results:")
    evaluate_model(rf, X_test, y_test)

    # SVM
    print("\nTraining SVM model...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    print("SVM Results:")
    evaluate_model(svm, X_test, y_test)

#  Evaluate a single model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred

# Visualize results
def visualize_results(model, X_test, y_test, column_names):

    # Feature Importance
    if isinstance(model, RandomForestClassifier):

        feature_importance = pd.Series(
            model.feature_importances_,
            index=column_names[:-1]
        )

        feature_importance.sort_values(ascending=False, inplace=True)

        plt.figure(figsize=(8, 5))

        sns.barplot(
            x=feature_importance,
            y=feature_importance.index
        )

        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()

        plt.savefig("feature_importance.png")
        plt.close()

    # Confusion Matrix
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Both figures saved successfully.")
# Main function to run the project
def main():
    # Load our dataset
    data = load_dataset()

    #  Explore the info
    explore_dataset(data)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train and test models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Visualize results (using Random Forest)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    visualize_results(rf, X_test, y_test, column_names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

# Run the project
if __name__ == "__main__":
    main()