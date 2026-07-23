import os
from flask import Flask, request, render_template
from joblib import load
import pandas as pd
from google import genai

app = Flask(__name__)

# 1. Safely fetch the API key from environment variables for secure deployment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    ai_client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    ai_client = None

# 2. Load trained model and symptoms
ml_model = load("saved_model/decision_tree.joblib")
df_train = pd.read_csv("dataset/training_data.csv")
symptoms = df_train.columns[:-2].tolist() 


@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms)


@app.route("/predict", methods=["POST"])
def predict():
    # Get selected symptoms from HTML form
    selected = request.form.getlist("symptoms")
    input_data = [1 if symptom in selected else 0 for symptom in symptoms]

    # Create DataFrame and predict disease
    input_df = pd.DataFrame([input_data], columns=symptoms)
    predicted_disease = ml_model.predict(input_df)[0]
    
    # 3. GenAI Integration using the correct model version
    ai_explanation = ""
    if ai_client:
        try:
            prompt = f"""
            You are a helpful, empathetic AI medical assistant. 
            A predictive machine learning model has suggested the user might have '{predicted_disease}' 
            based on these symptoms: {', '.join(selected)}. 
            Write a short, comforting paragraph (3-4 sentences) explaining what this condition generally is. 
            Remind the user that you are an AI, this is not a definitive diagnosis, and they should consult a doctor.
            """
            response = ai_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            ai_explanation = response.text
        except Exception as e:
            ai_explanation = "We couldn't generate an AI explanation right now, but please consult a doctor regarding this predicted condition."
    else:
        ai_explanation = "Please configure your GOOGLE_API_KEY environment variable on Render to enable AI explanations."

    return render_template("result.html", 
                           disease=predicted_disease, 
                           selected=selected, 
                           explanation=ai_explanation)


if __name__ == "__main__":
    app.run(debug=True)