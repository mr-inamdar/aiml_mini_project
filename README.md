# RideWise — Hourly Bike Rental Demand Forecasting

**AIML Mini Project | Third-Year B.E. / B.Tech Engineering**

RideWise is a machine learning web application that predicts how many bikes will be rented during a given hour at a city bike-sharing station. It is trained on two years of real-world rental data from Washington D.C. and served through a simple web interface built with Python Flask.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Dataset](#2-dataset)
3. [Input Features Used](#3-input-features-used)
4. [Machine Learning Algorithms](#4-machine-learning-algorithms)
5. [Prediction Workflow](#5-prediction-workflow)
6. [Project Architecture](#6-project-architecture)
7. [File-by-File Guide](#7-file-by-file-guide)
8. [Windows Setup Instructions](#8-windows-setup-instructions)
9. [Model Evaluation Results](#9-model-evaluation-results)
10. [Limitations of Predictions](#10-limitations-of-predictions)
11. [API Reference](#11-api-reference)
12. [Viva Questions and Answers](#12-viva-questions-and-answers)

---

## 1. Project Objective

City bike-sharing systems face a daily challenge: too many bikes at one station and too few at another. If station managers could estimate demand in advance, they could redistribute bikes before a shortage happens.

**The goal of RideWise** is to predict the total number of bikes (`cnt`) that will be rented during a specific hour using information that is known *before* that hour begins — the season, month, hour of day, day of week, holiday flag, weather condition, temperature, humidity, and wind speed.

This is a **supervised regression** problem because:
- The output (`cnt`) is a continuous number, not a category.
- The model is trained on historical examples where both the inputs and the correct answer (actual rentals) are known.

---

## 2. Dataset

| Property | Value |
| :--- | :--- |
| **Name** | UCI Bike Sharing Dataset |
| **Source** | [archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| **File used** | `hour.csv` |
| **Records** | 17,379 rows (one row per hour, Jan 2011 – Dec 2012) |
| **Columns** | 17 |
| **Missing values** | 0 |
| **Target column** | `cnt` (Total rentals = `casual` + `registered`) |
| **Target range** | 1 to 977 rentals per hour (mean ≈ 189) |

### How to download `hour.csv`

1. Visit: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
2. Click **Download** — a ZIP file will be saved.
3. Extract the ZIP. Inside you will find `hour.csv`.
4. Copy `hour.csv` into the `dataset\` folder inside the project:
   ```
   aiml_mini_project-main\
   └── dataset\
       └── hour.csv   ← place it here
   ```

> The `dataset\` folder already exists in the project. You only need to copy the file into it.

### Why only `hour.csv` and not `day.csv`?

The dataset comes with two files: `day.csv` (daily totals) and `hour.csv` (hourly totals). This project uses `hour.csv` because the web form predicts demand for a specific *hour*, and the `hr` column (hour of day) is one of the most important predictors.

---

## 3. Input Features Used

The model uses **10 input columns** from the dataset. These are the only columns available before the rental period begins.

| Feature | Column Name | Type | Allowed Values |
| :--- | :--- | :--- | :--- |
| Season | `season` | Integer | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| Month | `mnth` | Integer | 1 to 12 |
| Hour of day | `hr` | Integer | 0 to 23 |
| Day of week | `weekday` | Integer | 0=Sunday … 6=Saturday |
| Holiday | `holiday` | Integer | 0=No, 1=Yes |
| Working day | `workingday` | Integer | 1=Mon–Fri non-holiday, 0=Weekend or Holiday |
| Weather condition | `weathersit` | Integer | 1=Clear, 2=Mist, 3=Light Rain/Snow, 4=Heavy Rain |
| Temperature (normalized) | `temp` | Float | 0.0 to 1.0 (actual °C ÷ 41) |
| Humidity (normalized) | `hum` | Float | 0.0 to 1.0 (actual % ÷ 100) |
| Wind speed (normalized) | `windspeed` | Float | 0.0 to 1.0 (actual km/h ÷ 67) |

### Columns deliberately excluded

| Column | Reason for exclusion |
| :--- | :--- |
| `casual` | Sums into `cnt` — using it would be data leakage |
| `registered` | Sums into `cnt` — using it would be data leakage |
| `instant` | Row index, carries no real information |
| `dteday` | Calendar date, already captured by `season`, `mnth`, `weekday` |
| `yr` | Year indicator; excluded to keep the model general |
| `atemp` | "Feels like" temperature; highly correlated with `temp`, adds no new signal |

> **Data leakage** means accidentally giving the model information it would never have in a real prediction scenario. Since `casual + registered = cnt` exactly, including either would let the model "cheat".

---

## 4. Machine Learning Algorithms

Three regression algorithms were trained and compared on the same dataset split.

### 4.1 Linear Regression

Fits a straight-line relationship between the 10 input features and the rental count. Simple and fast, but cannot capture the non-linear patterns in bike demand (for example, demand spikes sharply at 8 AM and 5 PM but drops steeply after midnight — a straight line cannot represent this).

### 4.2 Decision Tree Regressor

Builds a tree of if-then rules by repeatedly splitting the data. For example: "If hour is between 16 and 19 AND it is a working day AND temperature > 0.5, then predict 380 bikes." Decision trees can capture non-linear patterns but tend to overfit (memorise training data) unless depth is limited.

**Settings used:** `max_depth=15`, `min_samples_split=5`, `random_state=42`

### 4.3 Random Forest Regressor *(Selected best model)*

A Random Forest builds **100 decision trees**, each trained on a random subset of the training data and a random subset of features. The final prediction is the average of all 100 trees. This averaging process reduces overfitting and produces more stable, accurate predictions.

**Settings used:** `n_estimators=100`, `max_depth=20`, `min_samples_split=4`, `random_state=42`

### Why Random Forest won

Random Forest achieved the lowest error (MAE = 86.58 bikes, RMSE = 124.30 bikes) and the highest R² score (0.6822) on the held-out test set. See Section 9 for the full comparison table.

---

## 5. Prediction Workflow

This diagram shows the complete path from raw data to a prediction displayed on screen.

```
hour.csv (17,379 rows)
        │
        ▼
 [train_model.py]
   1. Load & validate data
   2. Chronological 80/20 split
      ├── Train set: rows 1 – 13,903  (Jan 2011 – ~Oct 2012)
      └── Test set:  rows 13,904 – 17,379  (~Oct 2012 – Dec 2012)
   3. Preprocessing Pipeline (ColumnTransformer):
      ├── Categorical columns (season, mnth, hr, weekday, weathersit)
      │     → OneHotEncoder (converts each category to 0/1 columns)
      └── Numeric + Binary columns (temp, hum, windspeed, holiday, workingday)
            → Passthrough (used as-is)
   4. Train 3 models → compare MAE, RMSE, R²
   5. Save best model (Random Forest) → model/bike_model.joblib
   6. Save artifacts → metrics.json, feature_schema.json, feature_importance.json
        │
        ▼
 [app.py — Flask server]
   Loads bike_model.joblib once at startup
        │
        ▼
 [Browser — index.html]
   User fills the form (season, hr, temp, …) → clicks "Predict Demand"
        │
        ▼
 [main.js — POST /api/predict]
   Sends 10 feature values as JSON to Flask
        │
        ▼
 [app.py — /api/predict route]
   1. Validates all 10 inputs (type, range, NaN check)
   2. Calls predict_demand() in model/predict.py
   3. Model pipeline: preprocesses → predicts → clamps to ≥ 0 integer
   4. Returns { "status": "success", "predicted_demand": 712 }
        │
        ▼
 [Browser — main.js reads response]
   Displays "712 bikes / hour" in the result box
```

### What OneHotEncoder does

The `hr` column has values 0–23. The model cannot treat these as raw numbers because hour 23 is not "more" than hour 1 in any meaningful way — it is just a different category. OneHotEncoder converts `hr=17` into 24 binary columns where only the column `hr_17` is set to 1 and all others are 0. The same is applied to `season`, `mnth`, `weekday`, and `weathersit`.

---

## 6. Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (Client)                     │
│                                                             │
│  templates/index.html  ← renders the form and result box   │
│  static/css/style.css  ← visual styling                    │
│  static/js/main.js     ← form handling, API calls          │
└────────────────────┬────────────────────────────────────────┘
                     │  HTTP (GET / POST JSON)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Flask Web Server (app.py)                 │
│                                                             │
│  GET  /              → serve index.html                     │
│  GET  /api/health    → model status check                   │
│  GET  /api/metrics   → benchmark results from metrics.json  │
│  GET  /api/analytics → hourly/seasonal trends from CSV      │
│  POST /api/predict   → validate inputs → run inference      │
└────────────────────┬────────────────────────────────────────┘
                     │  Python function calls
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               ML Inference Layer (model/predict.py)         │
│                                                             │
│  load_artifacts()    → loads bike_model.joblib (once)       │
│  predict_demand()    → aligns columns, calls pipeline       │
└────────────────────┬────────────────────────────────────────┘
                     │  joblib.load()
                     ▼
┌─────────────────────────────────────────────────────────────┐
│             Serialized Pipeline (model/bike_model.joblib)   │
│                                                             │
│  Step 1: ColumnTransformer (OneHotEncoder + passthrough)    │
│  Step 2: RandomForestRegressor (100 trees)                  │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- The trained model is loaded **once** at server startup and kept in memory. Each prediction request does not reload the file — this keeps the API fast.
- All input validation happens in `app.py` *before* the model is called. If a value is missing, out of range, or NaN, the API returns a clear error message and never reaches the model.
- The ML code (`model/`) is completely separate from the web code (`app.py`, `templates/`, `static/`). The model can be retrained and re-saved without changing any web code.

---

## 7. File-by-File Guide

```
aiml_mini_project-main/
│
├── app.py                        ← Main Flask application
├── requirements.txt              ← Python package list
├── README.md                     ← This file
│
├── dataset/
│   └── hour.csv                  ← UCI Bike Sharing dataset (you must place this here)
│
├── model/
│   ├── train_model.py            ← Trains all 3 models, saves best one + artifacts
│   ├── predict.py                ← Reusable inference helper (load model, run prediction)
│   ├── retrain.py                ← Convenience script to re-run training
│   ├── bike_model.joblib         ← Serialized trained pipeline (generated by train_model.py)
│   ├── metrics.json              ← MAE/RMSE/R² results for all 3 models
│   ├── feature_schema.json       ← List of the 10 input features the model expects
│   └── feature_importance.json   ← Importance of each feature from Random Forest
│
├── templates/
│   └── index.html                ← Single-page web UI (form + result display)
│
├── static/
│   ├── css/style.css             ← All page styling
│   └── js/main.js                ← Form submission, API calls, result rendering
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py               ← 10 Flask API tests (routes, validation, errors)
│   ├── test_model.py             ← 6 model artifact and inference tests
│   └── test_dataset.py           ← 5 dataset integrity tests (columns, rows, no nulls)
│
└── docs/
    └── dataset_dictionary.md     ← Column-by-column explanation of hour.csv
```

### What each key file does in plain English

**`app.py`** — Starts the Flask web server. Defines 5 URL routes. When `/api/predict` receives a POST request, it validates the 10 input values, calls the model, and returns the predicted count as JSON. Also loads and caches the dataset analytics and model at startup.

**`model/train_model.py`** — Reads `hour.csv`, splits it 80/20 chronologically, builds a preprocessing pipeline, trains Linear Regression, Decision Tree, and Random Forest, prints a comparison table, and saves the best model to `bike_model.joblib`. Run this once before starting Flask.

**`model/predict.py`** — Provides two functions: `load_artifacts()` loads the saved `.joblib` file and schema, and `predict_demand()` accepts a dictionary of feature values, aligns them, runs the pipeline, and returns the prediction. Both `app.py` and the test suite call these functions.

**`model/bike_model.joblib`** — The serialized (saved-to-disk) scikit-learn Pipeline object. Contains the fitted OneHotEncoder and the trained Random Forest. Loaded with `joblib.load()` — no retraining needed to serve predictions.

**`templates/index.html`** — The single HTML page the user sees. Contains the input form (10 fields), the result box, the error box, and the metrics table. Uses Flask's `render_template` to be served.

**`static/js/main.js`** — Runs in the browser. Reads form values, sends them as JSON to `/api/predict`, and displays either the prediction or an error message. Also fetches `/api/health` and `/api/metrics` on page load.

**`requirements.txt`** — Lists the Python packages needed: `numpy`, `pandas`, `scikit-learn` (≥1.4.0), `joblib`, `Flask`, `pytest`.

---

## 8. Windows Setup Instructions

Follow these steps **in order** using **Command Prompt** (`cmd`) or **PowerShell**.

### Step 1 — Verify Python is installed

```cmd
python --version
```

You need Python 3.9 or higher. If this command fails, download Python from https://python.org and tick "Add Python to PATH" during installation.

### Step 2 — Navigate to the project folder

Replace the path below with wherever you unzipped the project:

```cmd
cd C:\Users\YourName\Desktop\aiml_mini_project-main
```

### Step 3 — Create a virtual environment

```cmd
python -m venv venv
```

This creates a folder called `venv\` that holds a private copy of Python and all packages, isolated from the rest of your computer.

### Step 4 — Activate the virtual environment

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

> If PowerShell shows an error about execution policy, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

After activation, your prompt will show `(venv)` at the start.

### Step 5 — Install all required packages

```cmd
pip install -r requirements.txt
```

This installs Flask, scikit-learn, pandas, numpy, joblib, and pytest. It may take 1–3 minutes.

### Step 6 — Place `hour.csv` in the dataset folder

```
aiml_mini_project-main\
└── dataset\
    └── hour.csv    ← must be here before training
```

### Step 7 — Train the model

```cmd
python model\train_model.py
```

**Expected output:**
```
======================================================================================
                            MODEL BENCHMARK RESULTS
======================================================================================
Model Name           | MAE        | RMSE       | R² Score   | Score (%)*    
--------------------------------------------------------------------------------------
Linear Regression    | 110.30     | 158.50     | 0.4832     | 55.66        %
Decision Tree        | 95.90      | 140.79     | 0.5923     | 61.45        %
Random Forest        | 86.58      | 124.30     | 0.6822     | 65.19        %
======================================================================================
*Note: Score (%) = max(0, 1 - MAE / mean(actual test values)) * 100 (Custom error-based score).
======================================================================================
```

This creates `model\bike_model.joblib`, `model\metrics.json`, `model\feature_schema.json`, and `model\feature_importance.json`.

### Step 8 — Start the Flask server

```cmd
python app.py
```

**Expected output:**
```
Starting RideWise Flask server on http://127.0.0.1:5000 ...
 * Running on http://127.0.0.1:5000
```

Open your browser and go to: **http://127.0.0.1:5000**

Press `Ctrl + C` in the terminal to stop the server.

### Step 9 — Run the automated tests (optional but recommended)

Open a **new** terminal window (with the venv activated), navigate to the project, and run:

```cmd
pytest tests\ -v
```

**Expected output:**
```
24 passed in ~7s
```

### Step 10 — Re-train the model (if needed later)

If you want to retrain from scratch at any time:

```cmd
python model\retrain.py
```

---

## 9. Model Evaluation Results

These results are read directly from `model\metrics.json`, which is generated by `model\train_model.py` after training on the actual `hour.csv` dataset. No results have been invented.

### Train / Test Split

| | Value |
| :--- | :--- |
| **Method** | Chronological split (no shuffling) |
| **Training set** | First 80% of rows — 13,903 hourly records |
| **Test set** | Last 20% of rows — 3,476 hourly records |

> A **chronological split** means the model is trained on older data and tested on newer data. This mimics real-world usage, where you always predict the future using a model trained on the past.

### Benchmark Comparison

| Algorithm | MAE (bikes) | RMSE (bikes) | R² Score | Score (%)* |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 110.30 | 158.50 | 0.4832 | 55.66% |
| Decision Tree | 95.90 | 140.79 | 0.5923 | 61.45% |
| **Random Forest** | **86.58** | **124.30** | **0.6822** | **65.19%** |

*\*Note: **Score (%)** is an explicitly defined error-based metric: `max(0, 1 - MAE / mean(actual test values)) * 100`. Standard classification accuracy does not apply to regression tasks; this metric indicates average relative demand precision against mean actual test volume (~248.8 bikes/hr).*

Random Forest was selected as the best model because it achieved the highest R² (0.6822), the lowest RMSE (124.30), the lowest MAE (86.58), and the highest prediction score (65.19%).

### What the metrics mean

**MAE (Mean Absolute Error) = 86.58 bikes**
On average, the model's prediction differs from the actual rental count by about 87 bikes per hour. For example, if the actual count is 300, the model might predict anywhere from ~213 to ~387.

**RMSE (Root Mean Squared Error) = 124.30 bikes**
Similar to MAE but penalises large errors more heavily. The model occasionally makes larger errors (for example, during unexpected events like sudden rain), and those pull RMSE above MAE.

**R² Score = 0.6822**
The model explains 68.22% of the total variation in hourly demand. A score of 1.0 would mean perfect prediction; 0.0 would mean the model is no better than always predicting the average (189 bikes). 0.68 is a reasonable result for a noisy real-world dataset.

### Feature Importance (from Random Forest)

These values come from `model\feature_importance.json` and show which features the Random Forest considered most useful.

| Feature | Aggregated Importance | What it means |
| :--- | :---: | :--- |
| `hr` (Hour of Day) | **46.18%** | The single most important factor — demand peaks at 8 AM and 5–6 PM on working days |
| `temp` (Temperature) | 19.74% | Higher temperature → more rentals (up to a point) |
| `hum` (Humidity) | 15.85% | Higher humidity → fewer rentals |
| `workingday` | 8.85% | Working days show a commute pattern; weekends show a leisure pattern |
| `mnth` (Month) | 2.34% | Summer months see higher demand than winter |
| `windspeed` | 2.16% | Strong winds reduce rentals |
| `weekday` | 1.94% | Which day of the week affects demand slightly |
| `weathersit` | 1.54% | Rain/snow reduces demand significantly |
| `season` | 1.25% | Partly captured by month and temperature |
| `holiday` | 0.16% | Holidays have a small but measurable effect |

---

## 10. Limitations of Predictions

Be honest about these when presenting the project:

1. **The model was trained on Washington D.C. data (2011–2012) only.** It may not generalise well to other cities with different infrastructure, culture, or climate.

2. **The model cannot account for unprecedented events.** A public concert, a road closure, a pandemic, or a city marathon will cause demand spikes or crashes that the model has no signal for.

3. **Weather inputs must be normalised by the user.** The form asks for `temp` between 0 and 1 (actual °C ÷ 41). A user who types the raw temperature in Celsius (e.g., 28 instead of 0.68) will get a validation error, not a prediction. This is by design, but it is not user-friendly for a general audience.

4. **R² = 0.68 means 32% of demand variation is unexplained.** The model captures the dominant patterns (rush hour, weather, season) but cannot explain all the randomness in rental behaviour.

5. **No real-time weather API is connected.** The user must look up and manually enter the current weather values. The model does not fetch weather data automatically.

6. **The training data ends in December 2012.** The model does not know about changes in the bike-sharing network size, new stations, population growth, or changes in fuel prices after 2012.

7. **Prediction is per hour, not per station.** The dataset represents total system-wide demand. The model cannot tell you which specific station needs more bikes.

---

## 11. API Reference

All endpoints are served by `app.py` on `http://127.0.0.1:5000`.

### GET `/api/health`

Returns whether the model is loaded and the dataset file is available.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "dataset_available": true,
  "timestamp": "2026-10-26T11:38:18.570889+00:00"
}
```

### GET `/api/metrics`

Returns the benchmark results for all three models from `metrics.json`.

### GET `/api/analytics`

Returns statistical summaries computed from `hour.csv`: hourly trends, weekday trends, seasonal trends, and weather trends.

### POST `/api/predict`

Accepts a JSON body with all 10 features, validates them, runs the trained model, and returns the prediction.

**Request body (all 10 fields are required):**
```json
{
  "season": 3,
  "mnth": 7,
  "hr": 17,
  "holiday": 0,
  "weekday": 3,
  "workingday": 1,
  "weathersit": 1,
  "temp": 0.72,
  "hum": 0.50,
  "windspeed": 0.20
}
```

**Successful response:**
```json
{
  "status": "success",
  "predicted_demand": 712,
  "raw_prediction": 711.53,
  "features_used": { ... }
}
```

**Error response (example — temp out of range):**
```json
{
  "status": "error",
  "message": "Field 'temp' must be between 0.0 and 1.0 (Normalized temperature (temp = celsius / 41)). Received: 1.45"
}
```

**Test the API from PowerShell (no browser needed):**
```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:5000/api/predict" `
  -ContentType "application/json" `
  -Body '{"season":3,"mnth":7,"hr":17,"holiday":0,"weekday":3,"workingday":1,"weathersit":1,"temp":0.72,"hum":0.50,"windspeed":0.20}'
```

---

## 12. Viva Questions and Answers

These questions are commonly asked in college project evaluations for machine learning projects. Read and understand each answer — do not memorise them word for word.

---

**Q1. What is regression? How is it different from classification?**

**A:** Regression predicts a **continuous numeric value**, such as the number of bikes that will be rented. Classification predicts a **category or label**, such as whether an email is spam or not spam.

In this project, we predict `cnt`, which can be any integer from 1 to 977. Since the output is a number on a continuous scale, it is a regression problem.

---

**Q2. What is the target variable in your project? Why is it called `cnt`?**

**A:** The target variable is `cnt`, which stands for "count" — the total number of bike rentals in a given hour. It is defined as:

`cnt = casual (unregistered users) + registered (member users)`

The model's job is to predict this single number given the 10 input features.

---

**Q3. What is data leakage, and how did you prevent it?**

**A:** Data leakage happens when you include information in your training features that would not actually be available at prediction time — especially information that is derived from or identical to the target.

In this dataset, `casual + registered = cnt` exactly. If we included `casual` or `registered` as input features, the model could simply add them together and always get a perfect answer. This would be cheating — in practice, we do not know how many casual or registered users will rent bikes *before* the hour happens.

We prevented leakage by explicitly excluding `casual`, `registered`, `instant`, `dteday`, and `yr` from the feature columns in `train_model.py`.

---

**Q4. What is a Random Forest? Why does it perform better than a single Decision Tree?**

**A:** A Random Forest is an **ensemble** of many decision trees. In this project, 100 trees are trained. Each tree:
- Is trained on a random sample of the training rows (with replacement — this is called *bootstrap sampling*).
- At each split, considers only a random subset of features.

The final prediction is the **average of all 100 trees' predictions**.

A single Decision Tree tends to overfit — it memorises the training data too closely and performs poorly on new data. By averaging 100 slightly different trees, the Random Forest reduces this overfitting. The random variation between trees means their errors partially cancel out, making the ensemble more accurate and stable.

---

**Q5. What is a Train-Test Split, and why did you use a chronological split instead of a random shuffle?**

**A:** A train-test split divides the dataset into two parts: one for training the model and one for evaluating it on unseen data.

We used an **80/20 chronological split** — the first 80% of rows (older data, Jan 2011 – ~Oct 2012) are used for training, and the last 20% (newer data, ~Oct 2012 – Dec 2012) are used for testing.

We did NOT shuffle the data randomly before splitting, because bike rental data is **time-series data**. If we randomly shuffled, future data points would end up in the training set and the model would essentially "see the future" during training. A chronological split correctly simulates the real-world scenario: train on the past, predict the future.

---

**Q6. What does MAE mean? What does an MAE of 86.58 mean for this project?**

**A:** MAE stands for **Mean Absolute Error**. It is calculated as the average of the absolute differences between predicted and actual values:

`MAE = (1/n) × Σ |actual − predicted|`

An MAE of 86.58 means that, on average, the model's prediction is off by about **87 bikes per hour**. For an hour where the actual demand is 300 bikes, the model might predict anywhere from roughly 213 to 387 bikes.

MAE is easy to interpret because it is in the same units as the target variable (number of bikes).

---

**Q7. What is RMSE and how is it different from MAE?**

**A:** RMSE stands for **Root Mean Squared Error**:

`RMSE = √( (1/n) × Σ (actual − predicted)² )`

The key difference is that RMSE **squares** the errors before averaging. This means large individual errors contribute much more to RMSE than to MAE. If the model makes a few very bad predictions (for example, predicting 100 bikes when 500 actually showed up), RMSE will be much higher than MAE.

In this project, RMSE (124.30) > MAE (86.58), which tells us the model occasionally makes some large errors, not just small consistent ones.

---

**Q8. What is R-squared (R²)? What does a value of 0.6822 mean?**

**A:** R-squared (also called the coefficient of determination) measures what **fraction of the total variation** in the target variable the model successfully explains. It ranges from 0 to 1 (and can be negative for very poor models).

- **R² = 1.0** → Perfect predictions. The model explains 100% of the variation.
- **R² = 0.0** → The model is no better than simply predicting the mean value every time.
- **R² = 0.6822** → The model explains **68.22%** of the variation in hourly bike demand. The remaining 31.78% is due to factors the model cannot capture (random events, data it was not given, etc.).

---

**Q9. What is OneHotEncoding? Why is it necessary?**

**A:** Machine learning models work with numbers. Columns like `hr` (0 to 23) look numeric but are actually **categorical** — hour 17 is not "17 times more" than hour 1. Treating them as raw integers would mislead the model.

OneHotEncoding converts each category into a separate binary (0 or 1) column. For example, `hr=17` becomes a row of 24 columns where column `hr_17=1` and all others are 0.

After encoding, the model can learn different coefficients or split rules for each hour independently, without imposing any false ordering.

---

**Q10. What is Flask, and why was it used here?**

**A:** Flask is a lightweight Python web framework. It lets you define URL routes and attach Python functions to them. When a browser sends a request to a URL, Flask runs the corresponding function and returns the result.

In RideWise, Flask serves the HTML page at `/` and handles prediction requests at `/api/predict`. It receives a JSON payload from the browser, runs the trained model, and sends the result back as JSON. Flask was chosen because it requires very few lines of code and no separate server setup — the server runs with a single `python app.py` command.

---

**Q11. What is Joblib, and why is it used instead of `pickle`?**

**A:** Joblib is a Python library for efficiently saving and loading large Python objects to disk. It is used to save the trained scikit-learn Pipeline to `model/bike_model.joblib`.

```python
joblib.dump(pipeline, "model/bike_model.joblib")   # saves
pipeline = joblib.load("model/bike_model.joblib")  # loads
```

Joblib is preferred over Python's built-in `pickle` for scikit-learn models because it handles large NumPy arrays (which make up the Random Forest's 100 decision trees) more efficiently — it compresses them and can use memory-mapped files.

---

**Q12. Why does the model use a scikit-learn Pipeline instead of preprocessing and training separately?**

**A:** A scikit-learn `Pipeline` chains preprocessing and the model into a single object:

```
Pipeline([
    ("preprocessor", ColumnTransformer(...)),
    ("regressor", RandomForestRegressor(...))
])
```

**Advantage:** When `pipeline.predict(X)` is called, it automatically applies the *same* preprocessing that was learned during training before passing data to the model. This eliminates the risk of applying different preprocessing at prediction time than was used during training — a common source of bugs.

It also means we can save the entire pipeline (including the fitted encoder) in one `.joblib` file and reload it without needing to re-fit anything.

---

**Q13. The model has an R² of only 0.68. Is that a good result?**

**A:** It depends on context. For a clean, synthetic dataset, 0.68 would be considered low. For real-world demand forecasting using only weather and calendar features — without GPS data, pricing data, competitor information, or event schedules — it is a reasonable result.

The dataset itself has high inherent variability. The maximum demand is 977 bikes/hour while the minimum is 1 bike/hour, and this spread is driven by factors beyond what our 10 input features capture. The model correctly learns the dominant patterns (rush-hour peaks, seasonal trends, weather effects) but cannot predict rare or unexpected events.

For a college mini-project, this result is fully acceptable and reflects honest machine learning practice.

---

**Q14. If a user enters temperature as 28 (Celsius) instead of 0.68, what happens?**

**A:** The server will return an HTTP 400 error with a clear message:

```json
{
  "status": "error",
  "message": "Field 'temp' must be between 0.0 and 1.0 (Normalized temperature (temp = celsius / 41)). Received: 28.0"
}
```

The validation happens in `validate_prediction_input()` in `app.py` before the model is ever called. This protects the model from receiving out-of-distribution inputs and gives the user an actionable error message.

The frontend (`main.js`) also validates `temp` client-side and would catch this before even sending the request.

---

**Q15. What would you do to improve this project further?**

**A:** Honest improvements would include:

- **Hyperparameter tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find better settings for the Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`). This could improve R² by a few percentage points.
- **Feature engineering:** Add `is_rush_hour` (1 if hr is 7–9 or 17–19), or `temp_squared` to capture the non-linear effect of extreme heat on demand.
- **Try gradient boosting:** Algorithms like `GradientBoostingRegressor` or XGBoost often outperform Random Forest on tabular data.
- **Live weather integration:** Connect to a weather API so the user does not need to normalise and enter weather values manually.
- **Cross-validation:** Instead of a single fixed split, use time-series cross-validation to get a more reliable estimate of model performance.

---

*RideWise — AIML College Mini Project | Built with Python, Flask, scikit-learn*
