/**
 * RideWise — Hourly Bike Rental Demand Forecasting
 * Frontend Logic (Vanilla JavaScript)
 * Synchronized with main.js for compatibility.
 */

document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    // Element references
    const statusBadge = document.getElementById("system-status-badge");
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    const resetBtn = document.getElementById("reset-btn");
    const loadingBox = document.getElementById("loading-indicator");
    const errorBox = document.getElementById("error-box");
    const errorMessage = document.getElementById("error-message");
    const resultBox = document.getElementById("prediction-result");
    const predictionCount = document.getElementById("prediction-count");
    const rawPredictionVal = document.getElementById("raw-prediction-val");
    const metricsContainer = document.getElementById("metrics-table-container");

    // =========================================================================
    // 1. Check Backend & Model Health Status
    // =========================================================================
    fetch("/api/health")
        .then((res) => {
            if (!res.ok) throw new Error("HTTP " + res.status);
            return res.json();
        })
        .then((data) => {
            if (data.status === "healthy" && data.model_loaded) {
                if (statusBadge) {
                    statusBadge.textContent = "● System Operational (Model Loaded)";
                    statusBadge.className = "badge badge-healthy";
                }
            } else {
                if (statusBadge) {
                    statusBadge.textContent = "▲ System Degraded";
                    statusBadge.className = "badge badge-degraded";
                }
            }
        })
        .catch(() => {
            if (statusBadge) {
                statusBadge.textContent = "▲ Backend Disconnected";
                statusBadge.className = "badge badge-error";
            }
        });

    // =========================================================================
    // 2. Fetch and Render Model Benchmark Metrics
    // =========================================================================
    fetch("/api/metrics")
        .then((res) => {
            if (!res.ok) throw new Error("HTTP " + res.status);
            return res.json();
        })
        .then((data) => {
            if (data && data.candidate_models && metricsContainer) {
                const championName = data.best_model ? data.best_model.name : "Random Forest";
                let html = `
                    <table class="benchmark-table">
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>Mean Absolute Error (MAE)</th>
                                <th>Root Mean Squared Error (RMSE)</th>
                                <th>R² Score</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                for (const [modelName, m] of Object.entries(data.candidate_models)) {
                    const isChampion = modelName === championName;
                    html += `
                        <tr class="${isChampion ? 'champion-row' : ''}">
                            <td>
                                <strong>${modelName}</strong>
                                ${isChampion ? '<span class="champion-tag">⭐ Selected Model</span>' : ''}
                            </td>
                            <td>${Number(m.mae).toFixed(2)}</td>
                            <td>${Number(m.rmse).toFixed(2)}</td>
                            <td>${Number(m.r2).toFixed(4)}</td>
                        </tr>
                    `;
                }

                html += `
                        </tbody>
                    </table>
                `;
                metricsContainer.innerHTML = html;
            }
        })
        .catch((err) => {
            console.warn("Could not load model metrics:", err);
            if (metricsContainer) {
                metricsContainer.innerHTML = '<p class="text-muted">Could not load benchmark metrics.</p>';
            }
        });

    // =========================================================================
    // 3. Client-side Validation Helper
    // =========================================================================
    function validateFormInputs(payload) {
        if (isNaN(payload.hr) || payload.hr < 0 || payload.hr > 23) {
            return "Hour must be an integer between 0 and 23.";
        }
        if (isNaN(payload.temp) || payload.temp < 0.0 || payload.temp > 1.0) {
            return "Temperature must be a normalized value between 0.0 and 1.0 (e.g., 0.70 for 28.7°C).";
        }
        if (isNaN(payload.hum) || payload.hum < 0.0 || payload.hum > 1.0) {
            return "Humidity must be a normalized value between 0.0 and 1.0 (e.g., 0.50 for 50%).";
        }
        if (isNaN(payload.windspeed) || payload.windspeed < 0.0 || payload.windspeed > 1.0) {
            return "Wind speed must be a normalized value between 0.0 and 1.0 (e.g., 0.20 for 13.4 km/h).";
        }
        return null;
    }

    // =========================================================================
    // 4. Prediction Form Submission Handler
    // =========================================================================
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();

            if (errorBox) errorBox.style.display = "none";
            if (resultBox) resultBox.style.display = "none";

            const payload = {
                season: parseInt(document.getElementById("season").value, 10),
                mnth: parseInt(document.getElementById("mnth").value, 10),
                hr: parseInt(document.getElementById("hr").value, 10),
                weekday: parseInt(document.getElementById("weekday").value, 10),
                workingday: parseInt(document.getElementById("workingday").value, 10),
                holiday: parseInt(document.getElementById("holiday").value, 10),
                weathersit: parseInt(document.getElementById("weathersit").value, 10),
                temp: parseFloat(document.getElementById("temp").value),
                hum: parseFloat(document.getElementById("hum").value),
                windspeed: parseFloat(document.getElementById("windspeed").value)
            };

            const validationError = validateFormInputs(payload);
            if (validationError) {
                if (errorMessage && errorBox) {
                    errorMessage.textContent = validationError;
                    errorBox.style.display = "block";
                }
                return;
            }

            if (predictBtn) {
                predictBtn.disabled = true;
                predictBtn.textContent = "Calculating...";
            }
            if (loadingBox) loadingBox.style.display = "flex";

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (response.ok && data.status === "success") {
                    if (predictionCount) predictionCount.textContent = data.predicted_demand.toLocaleString();
                    if (rawPredictionVal) rawPredictionVal.textContent = Number(data.raw_prediction).toFixed(2);
                    if (resultBox) resultBox.style.display = "block";
                } else {
                    if (errorMessage && errorBox) {
                        errorMessage.textContent = data.message || "Failed to generate prediction.";
                        errorBox.style.display = "block";
                    }
                }
            } catch (err) {
                if (errorMessage && errorBox) {
                    errorMessage.textContent = "Network error: Unable to communicate with the prediction backend (" + err.message + ").";
                    errorBox.style.display = "block";
                }
            } finally {
                if (predictBtn) {
                    predictBtn.disabled = false;
                    predictBtn.textContent = "Predict Demand";
                }
                if (loadingBox) loadingBox.style.display = "none";
            }
        });
    }

    // =========================================================================
    // 5. Reset Button Handler
    // =========================================================================
    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            document.getElementById("season").value = "3";
            document.getElementById("mnth").value = "7";
            document.getElementById("hr").value = "17";
            document.getElementById("weekday").value = "3";
            document.getElementById("workingday").value = "1";
            document.getElementById("holiday").value = "0";
            document.getElementById("weathersit").value = "1";
            document.getElementById("temp").value = "0.70";
            document.getElementById("hum").value = "0.50";
            document.getElementById("windspeed").value = "0.20";

            if (resultBox) resultBox.style.display = "none";
            if (errorBox) errorBox.style.display = "none";
            if (loadingBox) loadingBox.style.display = "none";
        });
    }
});
