import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("synthetic_loan_approval_dataset.csv")

# ลบคอลัมน์ที่ไม่จำเป็น
df.drop("Application_ID", axis=1, inplace=True)

# แยกข้อมูล
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Encode Target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# แยกประเภทข้อมูล
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Numerical Pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

# Combine
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    probability=True,
    random_state=42
)

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", svm)
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("="*50)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("="*50)
print(classification_report(y_test, y_pred))
print("="*50)
print(confusion_matrix(y_test, y_pred))

# Save
joblib.dump(model, "svm_loan_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel Saved")