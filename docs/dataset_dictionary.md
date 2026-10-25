# RideWise — Dataset Data Dictionary

## Overview
- **Dataset:** Bike Sharing Dataset (UCI Machine Learning Repository)
- **Primary File:** `dataset/hour.csv`
- **Total Records:** 17,379 hourly observations
- **Total Attributes:** 17 features
- **Missing Values:** None (0 null values across all columns)
- **Duplicate Records:** None

---

## Attribute Breakdown

### 1. Identification & Date
| Column Name | Data Type | Description | Values / Format |
| :--- | :--- | :--- | :--- |
| `instant` | Integer | Unique record index identifier | 1 to 17379 |
| `dteday` | String (Date) | Date of observation | `YYYY-MM-DD` (2011-01-01 to 2012-12-31) |

### 2. Temporal & Calendar Features
| Column Name | Data Type | Description | Values / Encoding |
| :--- | :--- | :--- | :--- |
| `season` | Integer | Season of the year | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| `yr` | Integer | Year indicator | 0: 2011, 1: 2012 |
| `mnth` | Integer | Month of the year | 1 to 12 |
| `hr` | Integer | Hour of the day | 0 to 23 (24-hour format) |
| `holiday` | Integer | Whether the day is an official public holiday | 0: No, 1: Yes (extracted from DC holiday schedule) |
| `weekday` | Integer | Day of the week | 0: Sunday, 1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday |
| `workingday` | Integer | Indicates if the day is neither weekend nor holiday | 1: Working day (Mon-Fri non-holiday), 0: Weekend or Holiday |

### 3. Weather & Environmental Features
| Column Name | Data Type | Description | Values / Normalization |
| :--- | :--- | :--- | :--- |
| `weathersit` | Integer | Weather condition category | 1: Clear, Few clouds, Partly cloudy<br>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds<br>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds<br>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog |
| `temp` | Float | Normalized ambient temperature in Celsius | Values divided by 41 (max temp). Range: [0.02, 1.00] (approx. 0.8°C to 41°C) |
| `atemp` | Float | Normalized "feels-like" temperature in Celsius | Values divided by 50 (max feels-like). Range: [0.00, 1.00] |
| `hum` | Float | Normalized relative humidity | Values divided by 100. Range: [0.00, 1.00] |
| `windspeed` | Float | Normalized wind speed | Values divided by 67 (max wind speed). Range: [0.00, 0.85] |

### 4. Target & Rental Breakdown Counts
| Column Name | Data Type | Description | Statistical Summary |
| :--- | :--- | :--- | :--- |
| `casual` | Integer | Count of casual (unregistered) bike rentals in that hour | Min: 0, Max: 367, Mean: ~35.67 |
| `registered` | Integer | Count of registered member bike rentals in that hour | Min: 0, Max: 886, Mean: ~153.79 |
| `cnt` | Integer | **Target Variable**: Total bike rental count (`casual` + `registered`) | Min: 1, Max: 977, Mean: ~189.46 |

---

## Machine Learning Considerations
- **Target Variable:** `cnt` represents continuous hourly demand, making this a regression problem.
- **Leakage Prevention:** Features `casual` and `registered` strictly sum up to `cnt` (`cnt = casual + registered`). These columns are known only after rentals occur and must be excluded during model training to avoid data leakage.
- **Identifiers:** `instant` is an artificial row index and `dteday` is redundant once decomposed into `yr`, `mnth`, `weekday`, and `hr`. Both should be excluded from model inputs.
