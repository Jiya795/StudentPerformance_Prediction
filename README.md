This is a small project I built to predict how a student might perform based on things like study time, past failures, absences and their first two period grades. You fill in a form, and the app tells you whether the student is likely to pass or fail, and also gives a rough estimate of the final grade.


Live Link: https://studentperformance-prediction-1-sw7q.onrender.com


How it works

I used the student performance dataset (data.csv) and trained two models on it:

Logistic Regression to predict pass / fail
Linear Regression to predict the final grade

The inputs are scaled with a StandardScaler before going into the models. The trained models, scaler and feature list are saved as .pkl files, and the Flask app loads them when it starts. When you submit the form, the app arranges your inputs in the same order as the training data, scales them, and runs both models.

Some of the inputs used: studytime, failures, absences, G1, G2, internet, higher and a few more. The full list is in features.pkl.

Project files
app.py - the Flask app (routes and prediction logic)
model.py - trains the models and saves the pickle files
data.csv - dataset
clf_model.pkl, reg_model.pkl, scaler.pkl, features.pkl - saved model files
metrics.json - evaluation scores from training
templates/ and static/ - HTML, CSS
tests/ - a few pytest tests
Procfile, requirements.txt, .python-version - needed for deployment
