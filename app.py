import gradio as gr
import pandas as pd
import numpy as np
import joblib

model = joblib.load("GradientBoosting_1weights.pkl")

# Load saved encoders + columns (IMPORTANT)
encoders = joblib.load("encoders.pkl")
trained_cols = joblib.load("columns.pkl")

def predict_file(file):
    try:
        df = pd.read_csv(file.name)

        # Drop target if exists
        if 'deterioration_next_12h' in df.columns:
            df = df.drop(columns=['deterioration_next_12h'])

        # 🔥 HANDLE STRING COLUMNS
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

                df[col] = df[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # 🔥 HANDLE 'none' / missing / invalid values
        df = df.replace(['none', 'nan', ''], np.nan)
        df = df.fillna(0)

        # Ensure column order
        df = df[trained_cols]

        probs = model.predict_proba(df)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            risk = prob * 100

            if risk >= 75:
                status = "⚠️ HIGH RISK"
            elif risk >= 40:
                status = "🟡 MODERATE RISK"
            else:
                status = "✅ STABLE"

            results.append(f"Patient {101+i}: {status} ({risk:.2f}%)")

        return "\n".join(results)

    except Exception as e:
        return f"Error: {str(e)}"


interface = gr.Interface(
    fn=predict_file,
    inputs=gr.File(label="Upload test.csv"),
    outputs=gr.Textbox(label="Prediction Results"),
    title="Clinical Deterioration Prediction System"
)

if __name__ == "__main__":
    interface.launch()