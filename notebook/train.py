import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from mlflow.data.pandas_dataset import from_pandas
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR,"..", "data", "wine_equilibre.csv")

mlflow.set_experiment("wine")

df = pd.read_csv(DATA_PATH)


df = df.rename(columns={
    "fixed acidity":"fixed_acidity",
    "volatile acidity":"volatile_acidity",
    "citric acid":"citric_acid",
    "residual sugar":"residual_sugar",
    "free sulfur dioxide":"free_sulfur_dioxide",
    "total sulfur dioxide":"total_sulfur_dioxide"
})

X = df.drop("quality", axis=1)
X["type"] = X["type"].apply(lambda x: 1 if x=="white" else 0)
df["quality"] = df["quality"].apply(lambda x: 1 if x=="Legit" else 0)
y = df["quality"]


num_cols = [col for col in X.columns if col != "type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", "passthrough", ["type"])
    ]
)

X_scaled = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

model_rfc = RandomForestClassifier()
model_dtc = DecisionTreeClassifier()
model_lr = LogisticRegression()
model_svc = SVC()

accuracie = []

dataset = from_pandas(df, source="wine_quality.csv", name="wine_fraud_data")

with mlflow.start_run():
    mlflow.log_input(dataset, context="training")

    # Random forest classifier
    model_rfc.fit(X_train, y_train)
    acc = model_rfc.score(X_test, y_test)
    mlflow.log_metric("accuracy rfc", acc)
    mlflow.sklearn.log_model(model_rfc, name="model_rfc")
    accuracie.append(acc)

    # Decision tree classifier
    model_dtc.fit(X_train, y_train)
    acc_dtc = model_dtc.score(X_test, y_test)
    mlflow.log_metric("accuracy dtc", acc_dtc)
    mlflow.sklearn.log_model(model_dtc, name="model_dtc")
    accuracie.append(acc_dtc)

    # Logistic regression
    model_lr.fit(X_train, y_train)
    acc_lr = model_lr.score(X_test, y_test)
    mlflow.log_metric("accuracy lr", acc_lr)
    mlflow.sklearn.log_model(model_lr, name="model_lr")
    accuracie.append(acc_lr)

    # svc
    model_svc.fit(X_train, y_train)
    acc_svc = model_svc.score(X_test, y_test)
    mlflow.log_metric("accuracy svc", acc_svc)
    mlflow.sklearn.log_model(model_svc, name="model_svc")
    accuracie.append(acc_svc)

    mlflow.sklearn.log_model(preprocessor, name="preprocessor")

    print("Accuracy:", accuracie)