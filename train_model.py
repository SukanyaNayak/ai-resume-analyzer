import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv(
    r"C:\Users\nayak\Downloads\Internship\ai_resume_screening.csv"
)

# =====================================================
# VIEW DATA
# =====================================================

print(data.head())
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["education_level"] = encoder.fit_transform(
    data["education_level"]
)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = data.drop("shortlisted", axis=1)

y = data["shortlisted"]
y = y.map({
    "No": 0,
    "Yes": 1
})
# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# SCALER
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# =====================================================
# MODEL
# =====================================================

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

# =====================================================
# SAVE MODEL & SCALER
# =====================================================

joblib.dump(model, "logistic_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")