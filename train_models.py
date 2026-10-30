
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline


# --------------------------------------------------
# PATHS AND SETUP
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_feedback.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)
np.random.seed(42)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA)

required = [
    "feedback_text",
    "sentiment",
    "complaint_type",
    "priority"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

df = df.dropna(subset=required).copy()
df["feedback_text"] = df["feedback_text"].astype(str).str.strip()
df = df[df["feedback_text"] != ""]

print(f"Dataset loaded: {len(df)} records")

results = {}


# --------------------------------------------------
# TRADITIONAL ML: SENTIMENT AND COMPLAINT TYPE
# --------------------------------------------------

for target, filename in [
    ("sentiment", "sentiment_pipeline.pkl"),
    ("complaint_type", "complaint_model.pkl")
]:

    print(f"\nTraining {target} model...")

    X = df["feedback_text"]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=12000,
                min_df=2,
                strip_accents="unicode"
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1200,
                class_weight="balanced"
            )
        )
    ])

    # Hyperparameter tuning
    if target == "sentiment":
        param_grid = {
            "classifier__C": [0.1, 1, 10]
        }

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        pipeline = search.best_estimator_

        print("Best parameters:", search.best_params_)

    else:
        pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(y_test, predictions)

    results[target] = {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )
    }

    joblib.dump(pipeline, MODELS / filename)

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=pipeline.classes_
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=pipeline.classes_
    )

    display.plot(xticks_rotation=45)
    plt.title(f"{target.title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS / f"{target}_confusion_matrix.png")
    plt.close()

    print(
        f"{target}: Accuracy={accuracy:.4f}, "
        f"Weighted F1={f1:.4f}"
    )


# --------------------------------------------------
# ANN: PRIORITY PREDICTION
# --------------------------------------------------

print("\nTraining ANN priority model...")

X_train, X_test, y_train, y_test = train_test_split(
    df,
    df["priority"],
    test_size=0.2,
    random_state=42,
    stratify=df["priority"]
)

# Text features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1500,
    min_df=2
)

Xtr_text = vectorizer.fit_transform(
    X_train["feedback_text"]
).toarray().astype("float32")

Xte_text = vectorizer.transform(
    X_test["feedback_text"]
).toarray().astype("float32")


# Metadata features
cat_cols = [
    c for c in ["channel", "customer_type"]
    if c in df.columns
]

num_cols = [
    c for c in ["order_value", "days_as_customer"]
    if c in df.columns
]

metadata_cols = cat_cols + num_cols

if metadata_cols:

    transformers = []

    if cat_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, cat_cols)
        )

    if num_cols:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numerical", numerical_pipeline, num_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    meta_train = preprocessor.fit_transform(
        X_train[metadata_cols]
    )

    meta_test = preprocessor.transform(
        X_test[metadata_cols]
    )

    # Convert sparse matrices to dense arrays
    if hasattr(meta_train, "toarray"):
        meta_train = meta_train.toarray()

    if hasattr(meta_test, "toarray"):
        meta_test = meta_test.toarray()

    meta_train = np.asarray(meta_train, dtype="float32")
    meta_test = np.asarray(meta_test, dtype="float32")

    joblib.dump(
        preprocessor,
        MODELS / "priority_preprocessor.pkl"
    )

else:

    preprocessor = None

    meta_train = np.empty(
        (len(X_train), 0),
        dtype="float32"
    )

    meta_test = np.empty(
        (len(X_test), 0),
        dtype="float32"
    )


# Combine text and metadata
Xtr = np.concatenate(
    [Xtr_text, meta_train],
    axis=1
)

Xte = np.concatenate(
    [Xte_text, meta_test],
    axis=1
)

# Encode priority labels
encoder = LabelEncoder()

ytr = encoder.fit_transform(y_train)
yte = encoder.transform(y_test)

joblib.dump(
    vectorizer,
    MODELS / "priority_vectorizer.pkl"
)

joblib.dump(
    encoder,
    MODELS / "priority_label_encoder.pkl"
)


# --------------------------------------------------
# BUILD ANN
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    Xtr,
    ytr,
    validation_split=0.15,
    epochs=15,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# ANN EVALUATION
# --------------------------------------------------

probabilities = model.predict(
    Xte,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

precision, recall, f1, _ = precision_recall_fscore_support(
    yte,
    predictions,
    average="weighted",
    zero_division=0
)

accuracy = accuracy_score(yte, predictions)

results["priority_ann"] = {
    "accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(f1),
    "classification_report": classification_report(
        yte,
        predictions,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )
}

model.save(MODELS / "priority_ann.keras")

# ANN confusion matrix
cm = confusion_matrix(yte, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot(xticks_rotation=45)
plt.title("Priority ANN Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS / "priority_confusion_matrix.png")
plt.close()

print(
    f"Priority ANN: Accuracy={accuracy:.4f}, "
    f"Weighted F1={f1:.4f}"
)


# --------------------------------------------------
# SAVE EVALUATION REPORT
# --------------------------------------------------

with open(
    REPORTS / "evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

with open(
    REPORTS / "evaluation_report.md",
    "w",
    encoding="utf-8"
) as file:

    file.write("# Customer Feedback Analyzer - Evaluation Report\n\n")
    file.write(
        "Evaluation uses stratified 80/20 train-test splits.\n\n"
    )

    for name, metrics in results.items():

        file.write(f"## {name}\n\n")
        file.write(
            f"- Accuracy: {metrics['accuracy']:.4f}\n"
        )
        file.write(
            f"- Weighted Precision: {metrics['weighted_precision']:.4f}\n"
        )
        file.write(
            f"- Weighted Recall: {metrics['weighted_recall']:.4f}\n"
        )
        file.write(
            f"- Weighted F1 Score: {metrics['weighted_f1']:.4f}\n\n"
        )


print("\n======================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("Models saved in models/")
print("Evaluation reports saved in reports/")
print("======================================")
