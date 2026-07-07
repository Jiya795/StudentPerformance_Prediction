"""
Basic test suite for the Student Performance Predictor Flask app.

Run with:
    pytest tests/ -v

These are intentionally lightweight (no mocking framework, no fixtures
beyond the Flask test client) — the goal is to catch obvious regressions
in routing, validation, and the prediction pipeline, not to be a
production-grade test suite.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


VALID_PAYLOAD = {
    'student_name': 'Test Student',
    'studytime': 3,
    'failures': 0,
    'absences': 4,
    'G1': 14,
    'G2': 15,
    'internet': 1,
    'higher': 1,
    'Medu': 3,
    'Fedu': 2,
    'health': 4,
    'famrel': 4,
    'freetime': 3,
}


def test_home_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Predict Your Academic Performance' in response.data


def test_dashboard_loads(client):
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'Data Dashboard' in response.data


def test_valid_form_prediction(client):
    response = client.post('/predict', data=VALID_PAYLOAD)
    assert response.status_code == 200
    assert b'Predicted Grade' in response.data


def test_form_prediction_missing_field_returns_400(client):
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload['G1']
    response = client.post('/predict', data=bad_payload)
    assert response.status_code == 400
    assert b"Missing value for" in response.data and b"G1" in response.data


def test_form_prediction_out_of_range_returns_400(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload['G1'] = 999
    response = client.post('/predict', data=bad_payload)
    assert response.status_code == 400
    assert b'must be between' in response.data


def test_api_predict_success(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != 'student_name'}
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['outcome'] in ('pass', 'fail')
    assert 0 <= data['grade'] <= 20
    assert 0 <= data['confidence'] <= 100
    assert isinstance(data['tips'], list) and len(data['tips']) > 0


def test_api_predict_invalid_json_returns_400(client):
    response = client.post('/api/predict', data='not json',
                            content_type='application/json')
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_api_predict_out_of_range_returns_400(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != 'student_name'}
    payload['absences'] = -5
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    assert 'error' in response.get_json()
