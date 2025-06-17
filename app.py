from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("output/math_score_model.pkl")
encoders = {
    'gender': joblib.load("output/le_gender.pkl"),
    'race_ethnicity': joblib.load("output/le_race_ethnicity.pkl"),
    'parental_level_of_education': joblib.load("output/le_parental_level_of_education.pkl"),
    'lunch': joblib.load("output/le_lunch.pkl"),
    'test_preparation_course': joblib.load("output/le_test_preparation_course.pkl")
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = {
            "gender": request.form["gender"],
            "race_ethnicity": request.form["race_ethnicity"],
            "parental_level_of_education": request.form["parental_level_of_education"],
            "lunch": request.form["lunch"],
            "test_preparation_course": request.form["test_preparation_course"],
            "reading_score": float(request.form["reading_score"]),
            "writing_score": float(request.form["writing_score"])
        }

        # Encode inputs
        X = [
            encoders["gender"].transform([data["gender"]])[0],
            encoders["race_ethnicity"].transform([data["race_ethnicity"]])[0],
            encoders["parental_level_of_education"].transform([data["parental_level_of_education"]])[0],
            encoders["lunch"].transform([data["lunch"]])[0],
            encoders["test_preparation_course"].transform([data["test_preparation_course"]])[0],
            data["reading_score"],
            data["writing_score"]
        ]

        # Make prediction
        prediction = model.predict([X])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
