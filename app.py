import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger('edupredict')

clf = pickle.load(open('clf_model.pkl', 'rb'))
reg = pickle.load(open('reg_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

try:
    with open('metrics.json') as f:
        METRICS = json.load(f)
except FileNotFoundError:
    
    METRICS = {}
    logger.warning('metrics.json not found — run `python model.py` to generate it.')

FIELD_RANGES = {
    'studytime': (1, 4),
    'failures': (0, 4),
    'absences': (0, 100),
    'G1': (0, 20),
    'G2': (0, 20),
    'internet': (0, 1),
    'higher': (0, 1),
    'Medu': (0, 4),
    'Fedu': (0, 4),
    'health': (1, 5),
    'famrel': (1, 5),
    'freetime': (1, 5),
}


def parse_and_validate(source):
    """Pull the model's input fields out of a form/JSON payload and validate
    them. Raises ValueError with a human-readable message on the first
    problem found, instead of letting an int() cast or a bad range blow up
    somewhere deep inside sklearn."""
    data = {}
    for field, (lo, hi) in FIELD_RANGES.items():
        raw = source.get(field, None)
        if raw is None or raw == '':
            raise ValueError(f"Missing value for '{field}'.")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{field}' must be a whole number, got {raw!r}.")
        if not (lo <= value <= hi):
            raise ValueError(f"'{field}' must be between {lo} and {hi}, got {value}.")
        data[field] = value
    return data


def get_tips(data):
    tips = []
    if data['studytime'] <= 2:
        tips.append("Increase your study time to at least 5 hours per week.")
    if data['absences'] > 10:
        tips.append("Reduce absences — attendance strongly affects performance.")
    if data['failures'] > 0:
        tips.append("Work on understanding past failed subjects with a tutor or peer.")
    if data['G1'] < 10 or data['G2'] < 10:
        tips.append("Focus on consistent performance in mid-term exams.")
    if data['health'] <= 2:
        tips.append("Take care of your health — sleep and exercise improve focus.")
    if data['internet'] == 0:
        tips.append("Access to internet resources can significantly boost learning.")
    if not tips:
        tips.append("Great profile! Keep up the consistent effort and study habits.")
    return tips


def run_prediction(data):
    """Shared by the HTML form route and the JSON API route so both stay
    in sync with exactly one prediction code path."""
    X = pd.DataFrame([[data[f] for f in features]], columns=features)
    X_scaled = scaler.transform(X)
    result = int(clf.predict(X_scaled)[0])
    grade = float(reg.predict(X_scaled)[0])
    grade = round(min(max(grade, 0), 20), 1)
    proba = clf.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)
    tips = get_tips(data)
    return {
        'result': result,
        'outcome': 'pass' if result == 1 else 'fail',
        'grade': grade,
        'confidence': confidence,
        'tips': tips,
    }


@app.route('/')
def index():
    return render_template('index.html', metrics=METRICS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        student_name = request.form.get('student_name', 'Student').strip() or 'Student'
        data = parse_and_validate(request.form)
    except ValueError as e:
        logger.info('Rejected form submission: %s', e)
        return render_template('error.html', message=str(e)), 400

    prediction = run_prediction(data)
    logger.info('Prediction served: outcome=%s grade=%s', prediction['outcome'], prediction['grade'])
    return render_template(
        'result.html',
        student_name=student_name,
        result=prediction['result'],
        grade=prediction['grade'],
        confidence=prediction['confidence'],
        tips=prediction['tips'],
        data=data
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for programmatic use — same validation and prediction logic
    as the HTML form, so integrations don't need a browser.

    Example:
        curl -X POST http://localhost:5000/api/predict \\
             -H "Content-Type: application/json" \\
             -d '{"studytime":3,"failures":0,"absences":4,"G1":14,"G2":15,
                  "internet":1,"higher":1,"Medu":3,"Fedu":2,"health":4,
                  "famrel":4,"freetime":3}'
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'Request body must be valid JSON.'}), 400
    try:
        data = parse_and_validate(payload)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    prediction = run_prediction(data)
    logger.info('API prediction served: outcome=%s grade=%s', prediction['outcome'], prediction['grade'])
    return jsonify({'input': data, **prediction})


@app.route('/dashboard')
def dashboard():
    try:
        df = pd.read_csv('data.csv')
        stats = {
            'total': len(df),
            'pass_rate': round((df['G3'] >= 10).mean() * 100, 1),
            'avg_grade': round(df['G3'].mean(), 1),
            'avg_study': round(df['studytime'].mean(), 1),
        }
    except FileNotFoundError:
        logger.warning('data.csv not found — using placeholder dashboard stats.')
        stats = {'total': 0, 'pass_rate': 0, 'avg_grade': 0, 'avg_study': 0}
    return render_template('dashboard.html', stats=stats, metrics=METRICS)


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, port=port)
