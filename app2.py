import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Multi-Disease Prediction", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv("multi_disease_dataset.csv")

# Load data
df = load_data()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model selection
model_name = st.selectbox(
    "Select Machine Learning Model",
    ["Random Forest", "Logistic Regression", "SVM", "Gradient Boosting"]
)

if model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=42)

elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_name == "SVM":
    model = SVC(probability=True)

elif model_name == "Gradient Boosting":
    model = GradientBoostingClassifier()

# Train model
model.fit(X_train, y_train)

disease_map = {0: "Healthy", 1: "Diabetes", 2: "Heart Disease"}

# UI
st.title("Healthcare Multi-Disease Prediction System")
st.subheader("Enter Patient Details")

input_data = []
for feature in X.columns:
    val = st.number_input(feature, value=0.0)
    input_data.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.success(f"Predicted Disease: {disease_map[prediction]}")

    st.write("Prediction Probabilities")
    for i, cls in enumerate(model.classes_):
        st.write(f"{disease_map[cls]}: {probabilities[i]*100:.2f}%")

    for i, cls in enumerate(model.classes_):
        if probabilities[i] > 0.7 and cls != 0:
            st.warning(f"High Risk of {disease_map[cls]}")

# -------- AUC SCORE AT THE END --------
st.markdown("---")
st.subheader(f"Overall Model Performance (AUC)")

auc_list = []

for cls in model.classes_:
    y_true = (y_test == cls).astype(int)
    y_prob = model.predict_proba(X_test)[:, cls]
    auc = roc_auc_score(y_true, y_prob)
    auc_list.append(auc)
    st.write(f"{disease_map[cls]} AUC: {auc:.2f}")

# Mean AUC
mean_auc = sum(auc_list) / len(auc_list)
st.success(f"Mean AUC Score ({model_name}): {mean_auc:.2f}")
