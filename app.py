from flask import Flask, render_template, request
import pandas as pd
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Create performance labels
def performance_label(g3):
    if g3 < 10:
        return "At Risk"
    elif g3 < 15:
        return "Average"
    else:
        return "High Performing"

df["performance"] = df["G3"].apply(performance_label)

# Features for prediction
features = ["studytime", "failures", "absences", "G1", "G2"]

X = df[features]
y = df["performance"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Ollama AI explanation function
def get_ai_explanation(prediction, confidence, risk_score, studytime, failures, absences, G1, G2):

    prompt = f"""
    You are an academic support assistant.

    A machine learning model predicted this student's performance level as: {prediction}.
    The model confidence is {confidence}%.
    The student's risk score is {risk_score} out of 100.

    Student information:
    - Study time level: {studytime} out of 4
    - Past failures: {failures}
    - Absences: {absences}
    - First period grade: {G1} out of 20
    - Second period grade: {G2} out of 20

    Write a short explanation in 3-5 sentences.
    Explain why the student may have received this prediction.
    Give one helpful academic recommendation.
    Keep the tone supportive and professional.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )

        data = response.json()
        return data.get("response", "No explanation generated.")

    except Exception:
        return "AI explanation unavailable. Make sure Ollama is running."

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    explanation = None
    confidence = None
    risk_score = None

    if request.method == "POST":

        studytime = int(request.form["studytime"])
        failures = int(request.form["failures"])
        absences = int(request.form["absences"])
        G1 = int(request.form["G1"])
        G2 = int(request.form["G2"])

        input_data = pd.DataFrame([[
            studytime,
            failures,
            absences,
            G1,
            G2
        ]], columns=features)

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]
        confidence = round(max(probabilities) * 100, 1)

        # Simple risk score calculation
        risk_score = 0
        risk_score += failures * 20
        risk_score += max(0, 10 - G1) * 3
        risk_score += max(0, 10 - G2) * 3
        risk_score += min(absences, 30)
        risk_score += max(0, 3 - studytime) * 5
        risk_score = min(100, risk_score)

        explanation = get_ai_explanation(
            prediction,
            confidence,
            risk_score,
            studytime,
            failures,
            absences,
            G1,
            G2
        )

    return render_template(
        "index.html",
        prediction=prediction,
        explanation=explanation,
        confidence=confidence,
        risk_score=risk_score
    )

if __name__ == "__main__":
    app.run(debug=True)