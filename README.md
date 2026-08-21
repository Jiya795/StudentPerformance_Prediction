Student Performance Predictor

A machine learning web application that predicts student academic performance and provides personalized improvement tips.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![Tests](https://img.shields.io/badge/Tests-pytest-blueviolet?style=flat)


What It Does
- Predicts whether a student will **Pass or Fail** their final exam
- Predicts the **exact final grade** (0–20 scale)
- Provides **personalized tips** based on the student's profile
- Shows an **analytics dashboard** with 6 charts, including a confusion matrix and a data-leakage comparison
- Exposes a **JSON API** (`/api/predict`) alongside the HTML form, for programmatic use

Model Performance
All numbers below are written to `metrics.json` by `model.py` on every training run — nothing here is hand-typed, so it can't go stale.

| Metric | Score |
|---|---|
| Classifier Accuracy (held-out test set) | **87%** |
| Classifier Precision / Recall / F1 | 0.92 / 0.84 / 0.88 |
| Regressor R² Score (held-out test set) | **0.85** |
| Training samples | 400 students |
| Test samples | 100 students |

Model selection — not just "we used Random Forest"
Three candidate models were cross-validated on identical folds for both tasks (Random Forest, Logistic/Linear Regression), and the winner by cross-validation score was deployed:

| Task | Winner | CV Score | Runner-up | CV Score |
|---|---|---|---|---|
| Classification | Logistic Regression | 92.3% acc | Random Forest (tuned) | 88.8% acc |
| Regression | Linear Regression | R² = 0.882 | Random Forest (tuned) | R² = 0.781 |

Random Forest was hyperparameter-tuned via `GridSearchCV` (`n_estimators`, `max_depth`, `min_samples_leaf`) before the comparison, so this isn't "tuned RF vs. default baseline" — it's a fair fight, and the simpler linear models won.

Data leakage disclosure
`G1` and `G2` (1st/2nd period grades) are extremely predictive of the final grade `G3` by construction — they're essentially earlier measurements of the same thing you're trying to predict. This is a known property of the underlying [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

| | With G1/G2 | Without G1/G2 |
|---|---|---|
| Classifier CV accuracy | 87.0% | 63.8% |
| Regressor CV R² | 0.805 | 0.214 |

The deployed model **keeps** G1/G2 because the product intentionally asks users for their in-progress grades — that's a reasonable, honest use case. But reporting one accuracy number without this context would overstate how much the model learns from behavioural/demographic features alone. Run `python model.py` to regenerate this comparison yourself; see `metrics.json` → `leakage_analysis`.

Tech Stack
| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | Scikit-learn — Logistic/Linear Regression (deployed), Random Forest (compared, tuned via GridSearchCV) |
| Validation | Stratified/K-Fold cross-validation, train/test split, confusion matrix |
| Data | Pandas, NumPy |
| Charts | Matplotlib, Seaborn |
| Frontend | HTML, Tailwind CSS |
| Testing | Pytest |
| Deployment | Render.com (Gunicorn) |

Project Structure
```
student_performance/
├── app.py              
├── model.py             
├── data.csv             
├── metrics.json          
├── clf_model.pkl         
├── reg_model.pkl         
├── scaler.pkl           
├── features.pkl          
├── requirements.txt      
├── tests/
│   └── test_app.py     
├── templates/
│   ├── base.html        
│   ├── index.html        
│   ├── result.html        
│   ├── error.html         
│   └── dashboard.html     
└── static/
    └── charts/          
```

Run Locally

```bash

git clone https://github.com/Jiya795/student-performance-predictor.git
cd student-performance-predictor


pip install -r requirements.txt


python model.py


python app.py
```
Open `http://localhost:5000` in your browser.

Set `FLASK_DEBUG=1` to run with Flask's debug reloader locally; it's off by default so it's safe to leave `app.py` as the entry point in any environment.

JSON API
In addition to the HTML form, there's a JSON endpoint for scripts/integrations:

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"studytime":3,"failures":0,"absences":4,"G1":14,"G2":15,
          "internet":1,"higher":1,"Medu":3,"Fedu":2,"health":4,
          "famrel":4,"freetime":3}'
```

Returns:
```json
{
  "input": { "...": "echoed back" },
  "result": 1,
  "outcome": "pass",
  "grade": 15.2,
  "confidence": 91.4,
  "tips": ["Great profile! Keep up the consistent effort and study habits."]
}
```
Invalid or out-of-range fields return a `400` with a JSON `error` message instead of a stack trace — the same validation logic backs both the form and the API.

Testing
```bash
pip install pytest
pytest tests/ -v
```
Covers: page loads, a valid end-to-end prediction through the form, out-of-range input rejection, and the JSON API's happy and error paths.

Deploy on Render
1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Start Command**: `gunicorn app:app`
5. Deploy!

Features
- Real-time ML prediction (Pass/Fail + Grade), via form or JSON API
- Personalized improvement tips
- Interactive dashboard with 6 data visualizations, including confusion matrix and leakage analysis
- Server-side input validation with clean error messages (no raw stack traces)
- Model selection backed by cross-validation, not a single fixed algorithm
- Fully responsive design
- Instant predictions (no page reload)

Built By
Jiya Gupta — B.Tech CSE Student, SOA University Bhubaneswar
[LinkedIn](https://www.linkedin.com/in/jiya-gupta-99b567356/) · [GitHub](https://github.com/Jiya795)
