"""Titanic training pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def run_pipeline(plan, data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    warnings = []
    df = pd.read_csv(data_path)
    # posto je receno da je survived posmatrana kolona mozda je nepotrebno ali zbog konfigurabilnosti bolji je pristup. 
    target = str(plan.get("target") or "Survived")
    if target not in df.columns:
        raise ValueError(f"Target column not found in data: {target}")

    # izbacujemo targed it dataframe- a
    df = df.dropna(subset=[target]).copy()
    y = df[target].astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Target column must be binary: {target}")

    selected = plan.get("selected_features")
    if isinstance(selected, list) and selected:
        feature_columns = [str(feature) for feature in selected]
    else:
        feature_columns = [column for column in df.columns if column != target]

    dropped = plan.get("drop_features")
    if isinstance(dropped, list):
        dropped = {str(feature) for feature in dropped}
        feature_columns = [feature for feature in feature_columns if feature not in dropped]

    usable_features = []
    for feature in feature_columns:
        if feature == target:
            warnings.append(f"Ignored target column '{target}' as an input feature.")
        elif feature not in df.columns:
            warnings.append(f"Ignored missing feature '{feature}'.")
        elif feature not in usable_features:
            usable_features.append(feature)

    if not usable_features:
        raise ValueError("No usable feature columns were selected.")

    X = df[usable_features]
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        feature for feature in usable_features if feature not in numeric_features
    ]

    transformers = []
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=str(plan.get("numeric_imputation") or "median")))
    ]
    if plan.get("scaling"):
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))

    categorical_strategy = str(plan.get("categorical_imputation") or "most_frequent")
    if categorical_strategy == "constant":
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", categorical_imputer),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    model_name = str(plan.get("model") or "logistic_regression")
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
            actual_model_name = model_name
        except ImportError:
            warnings.append("xgboost is not installed; using gradient_boosting instead.")
            model = models["gradient_boosting"]
            actual_model_name = "gradient_boosting"
    elif model_name in models:
        model = models[model_name]
        actual_model_name = model_name
    else:
        warnings.append(f"Unsupported model '{model_name}' ignored; using logistic_regression.")
        model = models["logistic_regression"]
        actual_model_name = "logistic_regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(plan.get("test_size") or 0.2),
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "status": "trained",
        "data_path": str(data_path),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": usable_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "requested_model": model_name,
        "model": actual_model_name,
        "metrics": {
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        },
        "warnings": warnings,
    }
