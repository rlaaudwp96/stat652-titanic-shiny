"""Compare five Titanic survival classifiers with scikit-learn."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def load_data():
    train = pd.read_csv(ROOT / "train.csv")
    return train[FEATURES], train["Survived"]

def make_preprocessor():
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical = ["Sex", "Embarked"]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])

def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=652),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=652),
        "Naive Bayes": GaussianNB(),
    }

def evaluate(test_size=0.20, random_state=652):
    X, y = load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    results, fitted = [], {}
    for name, estimator in models().items():
        pipe = Pipeline([("features", make_preprocessor()), ("model", estimator)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_valid)
        probability = pipe.predict_proba(x_valid)[:, 1]
        results.append({"Model": name, "Accuracy": accuracy_score(y_valid, pred), "Precision": precision_score(y_valid, pred), "Recall": recall_score(y_valid, pred), "F1": f1_score(y_valid, pred), "ROC_AUC": roc_auc_score(y_valid, probability)})
        fitted[name] = pipe
    return pd.DataFrame(results).sort_values("ROC_AUC", ascending=False), fitted

if __name__ == "__main__":
    metrics, _ = evaluate()
    print(metrics.round(4).to_string(index=False))
