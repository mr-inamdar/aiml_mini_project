# RideWise — Bike Rental Demand Forecasting Using Machine Learning

A complete machine learning web application that predicts hourly bike rental demand based on meteorological conditions, calendar schedules, and temporal patterns.

Designed as an end-to-end college engineering mini project (AIML / Computer Science).

---

## 🎯 Project Overview & Objectives

Urban bike-sharing systems provide a sustainable, efficient mode of urban transportation. However, operational challenges arise when stations face bike shortages during peak commute hours or surplus during unfavorable weather. 

**RideWise** addresses this imbalance by predicting hourly rental demand (`cnt`) using supervised regression techniques trained on real-world historical data from the Capital Bikeshare system (Washington D.C., USA).

### Key Highlights
- **Real-World Machine Learning:** Built using the authentic UCI Bike Sharing Dataset.
- **Explainable Feature Modeling:** Accounts for diurnal rhythms (hour of the day), seasonal fluctuations, temperature, humidity, wind speed, and working day vs. weekend patterns.
- **Clean Local Architecture:** Python Flask micro-framework with vanilla HTML5/CSS3/JavaScript (no complex frontend frameworks like React or Node.js required).
- **Interactive Visualizations:** Dynamic demand trend charts powered by Chart.js.
- **Automated Validation:** Test suite powered by `pytest`.

---

## 📂 Project Structure

```text
RideWise/
├── dataset/
│   └── hour.csv              # UCI Bike Sharing hourly dataset (17,379 records)
├── docs/
│   └── dataset_dictionary.md # Complete attribute descriptions & data dictionary
├── model/
│   └── .gitkeep              # Directory for serialized ML models (.joblib)
├── static/
│   ├── css/                  # Styling & responsive design
│   └── js/                   # Vanilla JS logic & Chart.js integration
├── templates/
│   └── .gitkeep              # Flask Jinja2 HTML templates
├── tests/
│   ├── __init__.py
│   └── test_dataset.py       # Automated dataset schema and integrity tests
├── .gitignore                # Git ignore configuration
├── README.md                 # Project documentation and setup guide
└── requirements.txt          # Python project dependencies
```

---

## 📊 Dataset Information

- **Source:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **Primary File:** `dataset/hour.csv`
- **Total Records:** 17,379 hourly entries (spanning 2011 to 2012)
- **Features:** 17 columns
- **Target Variable:** `cnt` (Total hourly bike rentals = `casual` + `registered`)
- **Missing Values:** 0 null values
- **Duplicate Rows:** 0 duplicate rows

### Target Variable Summary (`cnt`)
| Metric | Value |
| :--- | :--- |
| **Count** | 17,379 |
| **Mean** | 189.46 rentals/hour |
| **Standard Deviation** | 181.39 |
| **Minimum** | 1 rental/hour |
| **25th Percentile** | 40 rentals/hour |
| **Median (50%)** | 142 rentals/hour |
| **75th Percentile** | 281 rentals/hour |
| **Maximum** | 977 rentals/hour |

> For a full breakdown of all 17 columns, refer to [docs/dataset_dictionary.md](docs/dataset_dictionary.md).

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Visualization** | Chart.js (CDN-delivered, no Node.js needed) |
| **Backend API** | Python 3, Flask |
| **Data Processing & ML** | NumPy, pandas, scikit-learn |
| **Model Persistence** | Joblib |
| **Testing** | pytest |

---

## 💻 Setup Instructions (Windows)

Follow these exact steps in **PowerShell** or **Command Prompt** to set up and run the project locally.

### 1. Open Terminal and Navigate to Project Directory
```powershell
cd c:\Users\HP\Desktop\Coding\python\AIML\RideWise
```

### 2. Create and Activate a Python Virtual Environment
Creating an isolated environment keeps dependencies clean and prevents version conflicts:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# (If using standard Command Prompt cmd.exe instead):
# .\venv\Scripts\activate.bat
```

> **Note on PowerShell Execution Policy:** If you encounter a script execution restriction, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Model Training & Benchmarking
```powershell
python model/train_model.py
```

### 5. Start the Flask Backend Server
```powershell
python app.py
```
The server will start on `http://127.0.0.1:5000`.

### 6. Run the Automated Test Suite (21 Tests)
Run all schema, model pipeline, and Flask API tests:
```powershell
pytest tests/ -v
```

---

## 📡 REST API Documentation

| Method | Endpoint | Description | Sample Output / Payload |
| :--- | :--- | :--- | :--- |
| **GET** | `/` | Web dashboard | HTML5 Web Interface |
| **GET** | `/api/health` | Service and model availability status | `{"status": "healthy", "model_loaded": true}` |
| **GET** | `/api/metrics` | Held-out benchmark metrics for all models | `{"best_model": {"name": "Random Forest", ...}}` |
| **GET** | `/api/analytics` | Statistical summaries from authentic dataset | `{"summary": {...}, "hourly_trends": [...]}` |
| **POST** | `/api/predict` | Real-time demand inference | `{"predicted_demand": 712, "status": "success"}` |

### Sample Prediction Request (PowerShell)
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/api/predict" `
  -ContentType "application/json" `
  -Body '{
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
  }'
```

### Sample Prediction Request (cURL)
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

## 📈 Model Benchmark Results (Held-Out Chronological Split)

Trained on first 80% (13,903 records) and evaluated on remaining 20% (3,476 records):

| Model Algorithm | MAE (Rentals) | RMSE (Rentals) | R² Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 110.30 | 158.50 | 0.4832 | Baseline |
| **Decision Tree Regressor** | 95.90 | 140.79 | 0.5923 | Non-linear tree |
| **Random Forest Regressor** | **86.58** | **124.30** | **0.6822** | **Selected Best Model** |

### Top Predictive Feature Drivers:
1. **Hour of Day (`hr`):** 46.18% cumulative importance (peak hours at 8:00 AM & 5:00–6:00 PM)
2. **Temperature (`temp`):** 19.74% importance
3. **Humidity (`hum`):** 15.85% importance
4. **Working Day (`workingday`):** 8.85% importance
5. **Month & Season (`mnth`, `season`):** 3.59% combined importance

---

## 🚀 Development Roadmap

- [x] **Phase 1: Workspace Setup & Dataset Verification**
  - Project directory structure created
  - `dataset/hour.csv` verified (17,379 rows, 17 columns, zero missing values)
  - Data dictionary documented in `docs/`
  - Automated integrity tests implemented with `pytest`
- [x] **Phase 2: Data Preprocessing & Model Training**
  - Target leakage prevention (`casual`, `registered`, `instant`, `dteday` excluded)
  - Chronological 80/20 train-test split implemented
  - Feature encoding via scikit-learn `ColumnTransformer` inside unified `Pipeline`
  - Benchmarked Linear Regression, Decision Tree, and Random Forest
  - Selected Random Forest (R² = 0.6822, MAE = 86.58, RMSE = 124.30)
  - Serialized pipeline to `model/bike_model.joblib`
  - Exported `metrics.json`, `feature_schema.json`, `feature_importance.json`
  - Reusable inference helper created in `model/predict.py`
  - Retraining entrypoint created in `model/retrain.py`
- [x] **Phase 3: Flask Backend Development**
  - Flask web server implementation (`app.py`)
  - Endpoints created: `GET /`, `GET /api/health`, `GET /api/metrics`, `GET /api/analytics`, `POST /api/predict`
  - Strict input validation rejecting NaN, Inf, out-of-bounds numbers, invalid categories
  - Model caching (no retraining per request)
  - Full automated API test suite in `tests/test_api.py` (21 total tests passing)
- [ ] **Phase 4: Frontend Development & Visualizations**
  - Modern, responsive HTML5/CSS3 interface
  - Interactive Chart.js visualizations (hourly commute curve, weekly distribution)
- [ ] **Phase 5: End-to-End Testing & Project Documentation**
  - Project report and viva preparation guide
