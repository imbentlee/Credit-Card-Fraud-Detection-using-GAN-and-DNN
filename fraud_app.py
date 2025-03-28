import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

# --- Load models ---
@st.cache_resource
def load_models():
    aug_dnn_model = tf.keras.models.load_model("aug_baseline_dnn_model.keras")
    lgbm_model = joblib.load("opti_lgbm_model.pkl")
    hybrid_dnn_model = tf.keras.models.load_model("opti_hybrid_dnn_model.keras")
    return aug_dnn_model, lgbm_model, hybrid_dnn_model

aug_dnn_model, lgbm_model, hybrid_dnn_model = load_models()

# --- UI ---
st.title("Credit Card Fraud Detection")
st.markdown("Upload a CSV with **30 feature columns**. Optional columns like `Time`, `Amount`, or `Class` will be used if available.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
model_choice = st.selectbox("Choose model", ["Augmented DNN", "Hybrid LGBM-DNN"])
thresh = st.slider("Prediction Threshold (Recommended 0.40-0.50)", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        df_input.index.name = "Transaction ID"  # Make index visible

        if df_input.shape[1] < 30:
            st.error("Uploaded CSV must have at least 30 feature columns.")
        else:
            df_features = df_input.drop(columns=["Class"], errors='ignore')
            X = df_features.iloc[:, :30].values.astype(np.float32)

            if model_choice == "Augmented DNN":
                probs = aug_dnn_model.predict(X).flatten()
            else:
                lgbm_probs = lgbm_model.predict_proba(X)[:, 1].reshape(-1, 1)
                X = np.hstack([X, lgbm_probs])
                probs = hybrid_dnn_model.predict(X).flatten()

            preds = (probs >= thresh).astype(int)

            # Add predictions to DataFrame
            results_df = df_input.copy()
            results_df["Fraud Probability"] = probs
            results_df["Predicted Class"] = preds

            # --- Display relevant columns ---
            display_cols = []
            for col in ["Time", "Amount"]:
                if col in results_df.columns:
                    display_cols.append(col)
            display_cols += ["Fraud Probability", "Predicted Class"]

            st.success("Prediction Complete.")
            st.subheader("📋 Full Results:")
            st.dataframe(results_df[display_cols])

            # --- Show detected frauds ---
            fraud_df = results_df[results_df["Predicted Class"] == 1]
            st.subheader(f"⚠️ Total Fraud Cases Detected: {len(fraud_df)}")
            st.dataframe(fraud_df[display_cols])

            # --- Evaluation if labels exist ---
            if "Class" in df_input.columns:
                st.subheader("📊 Evaluation Metrics (Based on Provided Class Labels)")
                y_true = df_input["Class"].values
                roc = roc_auc_score(y_true, probs)
                prec = precision_score(y_true, preds)
                rec = recall_score(y_true, preds)
                f1 = f1_score(y_true, preds)
                cm = confusion_matrix(y_true, preds)

                st.write(f"**ROC-AUC:** {roc:.4f}")
                st.write(f"**Precision:** {prec:.4f}")
                st.write(f"**Recall:** {rec:.4f}")
                st.write(f"**F1-Score:** {f1:.4f}")
                st.text("Confusion Matrix:")
                st.text(cm)
                st.text("Classification Report:\n" + classification_report(y_true, preds, digits=4))

            # --- Download CSV with index ---
            csv = results_df.to_csv(index=True).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")