import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset correctly
data = pd.read_csv("CustomerInteractionData.csv")


# Clean headers
data.columns = data.columns.str.strip().str.replace("\ufeff", "")

# Debug: print what Pandas sees
print("Columns detected:", data.columns.tolist())
print(data.head())



# Clean column names to avoid hidden spaces
data.columns = data.columns.str.strip()

st.title("Customer Interaction Classifier")

# Preprocess
X = data["CustomerInteractionRawText"]
y = data["AgentAssignedTopic"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Show evaluation
st.subheader("Model Performance")
y_pred = model.predict(X_test)
#st.text(classification_report(y_test, y_pred))
st.text(classification_report(y_test, y_pred, zero_division=0))


# User input
st.subheader("Try it yourself")
user_text = st.text_area("Enter customer complaint text:")
if user_text:
    user_vec = vectorizer.transform([user_text])
    prediction = model.predict(user_vec)[0]
    st.success(f"Predicted Topic: {prediction}")
