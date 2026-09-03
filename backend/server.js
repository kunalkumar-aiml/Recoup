/**
 * Recoup backend — Express API.
 *
 * Thin orchestration layer: receives events from the frontend/demo,
 * forwards them to the Python ML service (root-cause + bandit + policy),
 * and exposes the audit trail + evaluation results for the dashboard.
 */
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json());

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";
const EVALUATION_RESULTS_PATH = path.join(__dirname, "..", "evaluation", "results.json");

app.get("/api/health", async (req, res) => {
  try {
    const r = await axios.get(`${ML_SERVICE_URL}/health`);
    res.json({ backend: "ok", ml_service: r.data });
  } catch (err) {
    res.status(503).json({ backend: "ok", ml_service: "unreachable", error: err.message });
  }
});

// Forward a failed-payment event to the ML service for a decision
app.post("/api/decide", async (req, res) => {
  try {
    const r = await axios.post(`${ML_SERVICE_URL}/decide`, req.body);
    res.json(r.data);
  } catch (err) {
    res.status(502).json({ error: "ml-service call failed", detail: err.message });
  }
});

// Recent audit trail, proxied from the ML service
app.get("/api/audit", async (req, res) => {
  try {
    const limit = req.query.limit || 50;
    const r = await axios.get(`${ML_SERVICE_URL}/audit`, { params: { limit } });
    res.json(r.data);
  } catch (err) {
    res.status(502).json({ error: "ml-service call failed", detail: err.message });
  }
});

// Baseline vs ML-only vs Recoup evaluation results, for the scoreboard
app.get("/api/evaluation", (req, res) => {
  if (!fs.existsSync(EVALUATION_RESULTS_PATH)) {
    return res.status(404).json({
      error: "No evaluation results yet. Run: cd evaluation && python3 evaluate.py",
    });
  }
  const data = JSON.parse(fs.readFileSync(EVALUATION_RESULTS_PATH, "utf-8"));
  res.json(data);
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Recoup backend listening on http://localhost:${PORT}`);
  console.log(`Forwarding ML calls to ${ML_SERVICE_URL}`);
});
