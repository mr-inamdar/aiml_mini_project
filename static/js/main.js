// RideWise Frontend Logic
document.addEventListener("DOMContentLoaded", () => {
    // 1. Check API Health
    fetch("/api/health")
        .then(res => res.json())
        .then(data => {
            const badge = document.getElementById("system-status-badge");
            if (data.status === "healthy" && data.model_loaded) {
                badge.textContent = "● System Operational (Model Loaded)";
                badge.style.backgroundColor = "#16a34a";
            } else {
                badge.textContent = "▲ Model Unavailable";
                badge.style.backgroundColor = "#dc2626";
            }
        })
        .catch(() => {
            const badge = document.getElementById("system-status-badge");
            badge.textContent = "▲ Backend Disconnected";
            badge.style.backgroundColor = "#dc2626";
        });

    // 2. Fetch and Display Metrics
    fetch("/api/metrics")
        .then(res => res.json())
        .then(data => {
            const container = document.getElementById("metrics-table-container");
            if (data && data.candidate_models) {
                let html = `
                <table style="width:100%; border-collapse:collapse; margin-top:1rem; text-align:left;">
                    <thead>
                        <tr style="border-bottom: 2px solid #e2e8f0;">
                            <th style="padding:8px;">Model</th>
                            <th style="padding:8px;">MAE</th>
                            <th style="padding:8px;">RMSE</th>
                            <th style="padding:8px;">R² Score</th>
                        </tr>
                    </thead>
                    <tbody>`;
                for (const [name, metrics] of Object.entries(data.candidate_models)) {
                    const isBest = name === data.best_model.name;
                    html += `
                        <tr style="border-bottom: 1px solid #e2e8f0; ${isBest ? 'font-weight:bold; background:#f0fdf4;' : ''}">
                            <td style="padding:8px;">${name} ${isBest ? '⭐ (Best)' : ''}</td>
                            <td style="padding:8px;">${metrics.mae}</td>
                            <td style="padding:8px;">${metrics.rmse}</td>
                            <td style="padding:8px;">${metrics.r2}</td>
                        </tr>`;
                }
                html += `</tbody></table>`;
                container.innerHTML = html;
            }
        })
        .catch(() => {
            document.getElementById("metrics-table-container").textContent = "Unable to load metrics.";
        });

    // 3. Handle Form Submission
    const form = document.getElementById("prediction-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("predict-btn");
        btn.disabled = true;
        btn.textContent = "Calculating...";

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

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (res.ok && data.status === "success") {
                document.getElementById("prediction-count").textContent = data.predicted_demand;
                document.getElementById("prediction-result").style.display = "block";
            } else {
                alert("Error: " + (data.message || "Failed to generate prediction"));
            }
        } catch (err) {
            alert("Network error: " + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = "Generate Prediction";
        }
    });
});
