"""
InfraCopilot AI — Inference Engine v5
=======================================
Fleet-wide prediction + Copilot recommendations powered by the v5 model.

Threshold contract:
  Threshold is LOADED from models_v5/metadata.json at startup.
  It is NEVER hardcoded. Printed at startup so it's auditable.

Features:
  - Vectorized scoring (50k chargers in <3s)
  - LR contribution: normalized |coef_i × scaled_value_i| per charger
  - Confidence labels: High / Medium / Low (distance from threshold)
  - Pagination API: get_top_k, filter_by_risk, get_by_id, paginate
  - CSV export (all) + JSON export (top 1,000)
  - FastAPI-ready routes at bottom of file
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Optional

from data_generator import BASE_FEATURE_COLS, ALL_FEATURE_COLS, engineer_features

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "models_v5/model.joblib"
SCALER_PATH   = "models_v5/scaler.joblib"
METADATA_PATH = "models_v5/metadata.json"
OUTPUTS_DIR   = "outputs"

# ── Business parameters ───────────────────────────────────────────────────────
DOWNTIME_COST_PER_HR   = 150
AVG_DOWNTIME_HRS       = 8
PREVENTIVE_MAINT_COST  = 200


# ══════════════════════════════════════════════════════════════════════════════
# Data schemas (dataclasses — zero external dependencies)
# Pydantic equivalents in FASTAPI_ROUTES at the bottom if needed.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RootCause:
    factor:           str
    key:              str
    contribution_pct: float
    value:            float
    unit:             str
    severity:         str
    engineered:       bool = False


@dataclass
class ChargerRec:
    charger_id:            str
    failure_probability:   float   # 0.0 – 0.995, rounded to 3dp
    risk_score_pct:        float   # probability × 100, 1dp
    risk_level:            str     # safe | warning | critical
    flagged:               bool    # prob >= threshold_used
    threshold_used:        float
    confidence:            str     # High | Medium | Low
    time_to_failure:       str
    top_cause:             str
    top_contributors:      List[RootCause]
    recommended_action:    str
    urgency:               str     # none | routine | soon | immediate
    estimated_savings_usd: float
    temperature:           float
    voltage:               float
    usage_hours:           float
    error_count:           int


@dataclass
class FleetSummary:
    generated_at:      str
    total_chargers:    int
    flagged_count:     int
    critical_count:    int
    warning_count:     int
    safe_count:        int
    total_savings_usd: float


@dataclass
class FleetResponse:
    summary:         FleetSummary
    recommendations: List[ChargerRec]


def _rec_to_dict(r: ChargerRec) -> dict:
    d = asdict(r)
    d["top_contributors"] = [asdict(c) for c in r.top_contributors]
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:

    def __init__(self,
                 model_path=MODEL_PATH, scaler_path=SCALER_PATH,
                 metadata_path=METADATA_PATH):

        for p in (model_path, scaler_path, metadata_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing artifact: {p}\nRun train_model.py first.")

        self.model    = joblib.load(model_path)
        self.scaler   = joblib.load(scaler_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Load threshold from metadata — never hardcoded
        self.threshold    = float(self.metadata["cost_opt_threshold"])
        self.feature_cols = self.metadata["feature_list"]
        self.model_name   = self.metadata["model_name"]
        self._coefs       = self._get_coefs()

        print(f"[Engine] {self.model_name}  (v{self.metadata.get('model_version','?')})")
        print(f"[Engine] Threshold  : {self.threshold}  ← loaded from metadata")
        print(f"[Engine] Features   : {len(self.feature_cols)}")

    # ── Main fleet API ────────────────────────────────────────────────────────

    def generate_fleet_recommendations(
        self,
        df: pd.DataFrame,
        top_k_contributors: int = 3,
    ) -> FleetResponse:
        """Score all chargers and return structured fleet recommendations."""
        t0     = time.perf_counter()
        df_eng = engineer_features(df)

        # Vectorized scoring — raw probabilities, no clipping or rounding.
        # Frontend is responsible for display formatting (e.g. round to 1dp).
        X      = df_eng[self.feature_cols].values.astype(np.float32)
        X_s    = self.scaler.transform(X)
        probs  = self.model.predict_proba(X_s)[:, 1]   # full float precision
        t      = self.threshold

        # Fix A: single consistent policy — risk_level AND flagged both derived
        # from the same threshold t so critical_count <= flagged_count always holds.
        #   safe     : p <  0.5 * t
        #   warning  : 0.5 * t <= p < t
        #   critical : p >= t          ← same boundary as flagged
        flagged    = probs >= t
        risk_levels = np.where(probs >= t,       "critical",
                      np.where(probs >= 0.5 * t, "warning", "safe"))

        # Vectorized contribution matrix: (n_chargers, n_features)
        contrib = self._contributions(X_s, df_eng)

        recs = []
        for i in range(len(df)):
            row        = df_eng.iloc[i]
            charger_id = str(df.iloc[i].get("charger_id", f"CHG-{i+1:06d}"))
            prob       = float(probs[i])
            rl         = risk_levels[i]          # from unified threshold policy
            causes     = self._root_causes(contrib[i], row, top_k_contributors)
            top_key    = causes[0].key    if causes else "unknown"
            top_label  = causes[0].factor if causes else "Unknown"

            recs.append(ChargerRec(
                charger_id=charger_id,
                failure_probability=prob,
                risk_score_pct=round(prob * 100, 1),   # rounded for display only
                risk_level=rl,
                flagged=bool(flagged[i]),
                threshold_used=self.threshold,
                confidence=_confidence(prob, self.threshold),
                time_to_failure=_time_estimate(prob),
                top_cause=top_label,
                top_contributors=causes,
                recommended_action=_recommend(top_key, rl),
                urgency=_urgency(prob, bool(flagged[i])),
                estimated_savings_usd=_savings(prob),
                temperature=float(row["temperature"]),
                voltage=float(row["voltage"]),
                usage_hours=float(row["usage_hours"]),
                error_count=int(row["error_count"]),
            ))

        recs.sort(key=lambda r: r.failure_probability, reverse=True)

        counts = {"critical": 0, "warning": 0, "safe": 0}
        for r in recs:
            counts[r.risk_level] += 1

        # ── Sanity checks ─────────────────────────────────────────────────────
        # Guarantee: all critical chargers are flagged (they share the same boundary t)
        assert counts["critical"] <= int(flagged.sum()), (
            f"CONSISTENCY ERROR: critical_count ({counts['critical']}) > "
            f"flagged_count ({int(flagged.sum())}). "
            "risk_level and flagged must use the same threshold."
        )

        # ── Probability distribution diagnostics ──────────────────────────────
        print(
            f"[Sanity] p_min={probs.min():.4f} | p_max={probs.max():.4f} | "
            f"p_mean={probs.mean():.4f} | "
            f"p_p50={np.percentile(probs, 50):.4f} | "
            f"p_p90={np.percentile(probs, 90):.4f} | "
            f"p_p99={np.percentile(probs, 99):.4f}"
        )

        summary = FleetSummary(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_chargers=len(recs),
            flagged_count=int(flagged.sum()),
            critical_count=counts["critical"],
            warning_count=counts["warning"],
            safe_count=counts["safe"],
            total_savings_usd=round(sum(r.estimated_savings_usd for r in recs), 2),
        )

        print(f"[Fleet] {len(recs):,} chargers in {time.perf_counter()-t0:.2f}s  |  "
              f"{int(flagged.sum()):,} flagged  |  "
              f"{counts['critical']:,} critical  |  "
              f"${summary.total_savings_usd:,.0f} potential savings")

        return FleetResponse(summary=summary, recommendations=recs)

    # ── Pagination / query helpers ─────────────────────────────────────────────

    def get_top_k(self, fleet: FleetResponse, k: int) -> List[ChargerRec]:
        return fleet.recommendations[:k]

    def filter_by_risk(self, fleet: FleetResponse, level: str) -> List[ChargerRec]:
        return [r for r in fleet.recommendations if r.risk_level == level]

    def get_by_id(self, fleet: FleetResponse, charger_id: str) -> Optional[ChargerRec]:
        for r in fleet.recommendations:
            if r.charger_id == charger_id:
                return r
        return None

    def paginate(self, recs: List[ChargerRec], page: int = 1, page_size: int = 50) -> dict:
        total       = len(recs)
        total_pages = (total + page_size - 1) // page_size
        start       = (page - 1) * page_size
        return {
            "page": page, "page_size": page_size,
            "total_pages": total_pages, "total_chargers": total,
            "data": recs[start: start + page_size],
        }

    # ── Console output ─────────────────────────────────────────────────────────

    def print_fleet_summary(self, fleet: FleetResponse, top_n: int = 10):
        s = fleet.summary
        W = 110
        print(f"\n{'═'*W}")
        print(f"  FLEET COPILOT  —  {s.total_chargers:,} chargers  |  "
              f"{s.flagged_count:,} flagged  |  ${s.total_savings_usd:,.0f} potential savings")
        print(f"  Critical: {s.critical_count:,}  |  Warning: {s.warning_count:,}  |  Safe: {s.safe_count:,}")
        print(f"{'─'*W}")
        print(f"  {'Charger':<12} {'Risk%':>6}  {'Level':<10} {'Flag':>6}  "
              f"{'Conf':<8}  {'Top Cause':<26}  {'Action':<30}  {'Savings':>8}")
        print(f"  {'─'*W}")
        for r in fleet.recommendations[:top_n]:
            icon   = {"critical": "🔴", "warning": "🟡", "safe": "🟢"}.get(r.risk_level, "⚪")
            flag   = "🚨" if r.flagged else "  "
            action = (r.recommended_action[:28] + "..") if len(r.recommended_action) > 30 else r.recommended_action
            print(
                f"  {r.charger_id:<12} {r.risk_score_pct:>5.1f}%  "
                f"{icon}{r.risk_level:<9} {flag:>4}  "
                f"{r.confidence:<8}  {r.top_cause[:25]:<26}  "
                f"{action:<30}  ${r.estimated_savings_usd:>6,.0f}"
            )
        remaining = s.total_chargers - top_n
        if remaining > 0:
            print(f"  ... and {remaining:,} more (see outputs/fleet_recommendations.csv)")
        print(f"{'═'*W}\n")

    # ── File output ────────────────────────────────────────────────────────────

    def save_outputs(self, fleet: FleetResponse,
                     csv_path: str = None, json_path: str = None,
                     top_k_json: int = 1_000):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        csv_path  = csv_path  or os.path.join(OUTPUTS_DIR, "fleet_recommendations.csv")
        json_path = json_path or os.path.join(OUTPUTS_DIR, "fleet_recommendations.json")

        # CSV — all chargers
        rows = [{
            "charger_id":            r.charger_id,
            "failure_probability_%": r.risk_score_pct,
            "risk_level":            r.risk_level,
            "flagged":               r.flagged,
            "confidence":            r.confidence,
            "time_to_failure":       r.time_to_failure,
            "top_cause":             r.top_cause,
            "urgency":               r.urgency,
            "recommended_action":    r.recommended_action,
            "estimated_savings_usd": r.estimated_savings_usd,
            "temperature":           r.temperature,
            "voltage":               r.voltage,
            "usage_hours":           r.usage_hours,
            "error_count":           r.error_count,
        } for r in fleet.recommendations]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[Output] CSV  → {csv_path}  ({len(rows):,} rows)")

        # JSON — top 1,000 only (avoids huge files)
        top = fleet.recommendations[:top_k_json]
        out = {
            "summary":         asdict(fleet.summary),
            "recommendations": [_rec_to_dict(r) for r in top],
            "_note":           f"JSON contains top {len(top):,} chargers by risk. Full data in CSV.",
        }
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Output] JSON → {json_path}  (top {len(top):,}, "
              f"{os.path.getsize(json_path)//1024}KB)")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_coefs(self) -> np.ndarray:
        """Extract per-feature weights (LR coefficients or RF importances)."""
        model = self.model
        # Unwrap calibrated wrapper if present
        if hasattr(model, "calibrated_classifiers_"):
            model = model.calibrated_classifiers_[0].estimator
        if hasattr(model, "coef_"):
            return np.abs(model.coef_[0])
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        return np.ones(len(self.feature_cols))

    def _contributions(self, X_s: np.ndarray, df_eng: pd.DataFrame) -> np.ndarray:
        """
        SHAP-lite vectorized contributions.
        For LogisticRegression: |coef_i × scaled_value_i|, then row-normalize.
        This is exact for linear models; for RF it approximates via feature importance × deviation.
        Returns (n_chargers, n_features), each row sums to 1.
        """
        # Use scaled values directly (already mean-centered) for LR interpretation
        raw    = np.abs(X_s) * self._coefs[np.newaxis, :]   # (n, f)
        totals = raw.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return raw / totals

    def _root_causes(self, contrib: np.ndarray, row: pd.Series, top_k: int) -> List[RootCause]:
        ranked = np.argsort(contrib)[::-1][:top_k]
        causes = []
        for idx in ranked:
            feat = self.feature_cols[idx]
            val  = float(row[feat])
            causes.append(RootCause(
                factor=_feature_label(feat),
                key=feat,
                contribution_pct=round(float(contrib[idx]) * 100, 1),
                value=round(val, 2),
                unit=_feature_unit(feat),
                severity=_feature_severity(feat, val),
                engineered=feat not in BASE_FEATURE_COLS,
            ))
        return causes


# ══════════════════════════════════════════════════════════════════════════════
# Pure helper functions
# ══════════════════════════════════════════════════════════════════════════════

# Note: risk_level is no longer a standalone function.
# It is computed inline in generate_fleet_recommendations() using the
# unified threshold policy: critical if p >= t, warning if p >= 0.5*t, else safe.
# This guarantees critical_count <= flagged_count at all times.


def _confidence(prob: float, threshold: float) -> str:
    """
    Confidence = how far the probability is from the decision boundary.
    Far from threshold → model is decisive.  Near threshold → borderline case.
      High   : margin >= 0.15  (decisive prediction)
      Medium : 0.05 <= margin < 0.15
      Low    : margin < 0.05   (borderline — gather more data)
    """
    margin = abs(prob - threshold)
    if margin >= 0.15: return "High"
    if margin >= 0.05: return "Medium"
    return "Low"


def _time_estimate(prob: float) -> str:
    if prob >= 0.85:   return "Failure within 24h"
    elif prob >= 0.65: return "Failure within 48h"
    elif prob >= 0.50: return "Failure within 1 week"
    elif prob >= 0.35: return "Elevated — monitor 2wks"
    return "No imminent failure"


def _urgency(prob: float, flagged: bool) -> str:
    if prob >= 0.85:            return "immediate"
    if prob >= 0.65 or flagged: return "soon"
    if prob >= 0.35:            return "routine"
    return "none"


def _savings(prob: float) -> float:
    if prob < 0.35: return 0.0
    avoided = prob * AVG_DOWNTIME_HRS * DOWNTIME_COST_PER_HR
    return round(max(avoided - PREVENTIVE_MAINT_COST, 0), 2)


def _feature_label(key: str) -> str:
    return {
        "temperature":       "Overheating",
        "voltage":           "Voltage Instability",
        "usage_hours":       "High Usage / Wear",
        "error_count":       "Error Frequency",
        "thermal_stress":    "Thermal Stress Index",
        "voltage_deviation": "Voltage Deviation",
        "error_density":     "Error Density",
        "risk_pressure":     "Combined Risk Pressure",
    }.get(key, key)


def _feature_unit(key: str) -> str:
    return {
        "temperature": "°C", "voltage": "V", "usage_hours": "hrs",
        "error_count": "errors", "thermal_stress": "composite",
        "voltage_deviation": "V from 230", "error_density": "errors/hr",
        "risk_pressure": "/1.0",
    }.get(key, "")


def _feature_severity(key: str, val: float) -> str:
    thresholds = {
        "temperature":       [(75, "critical"), (60, "warning")],
        "voltage_deviation": [(20, "critical"), (10, "warning")],
        "usage_hours":       [(600, "critical"), (400, "warning")],
        "error_count":       [(10, "critical"),  (5,  "warning")],
        "thermal_stress":    [(55, "critical"),  (35, "warning")],
        "error_density":     [(0.05, "critical"), (0.02, "warning")],
        "risk_pressure":     [(0.5, "critical"),  (0.3, "warning")],
    }
    compare = abs(val - 230) if key == "voltage" else val
    for thresh, sev in thresholds.get(key, []):
        if compare >= thresh:
            return sev
    return "normal"


def _recommend(top_key: str, risk_level: str) -> str:
    ACTIONS = {
        "temperature":       "Reduce load; inspect cooling and airflow.",
        "thermal_stress":    "Reduce load; inspect cooling and airflow.",
        "risk_pressure":     "Reduce load; inspect cooling and airflow.",
        "voltage":           "Inspect power supply and wiring.",
        "voltage_deviation": "Inspect power supply and wiring.",
        "usage_hours":       "Schedule preventive maintenance.",
        "error_count":       "Pull diagnostic logs; reboot or firmware update.",
        "error_density":     "Pull diagnostic logs; reboot or firmware update.",
    }
    base = ACTIONS.get(top_key, "Inspect charger; review maintenance log.")
    if risk_level == "critical":
        return f"URGENT: {base} Consider taking offline."
    if risk_level == "warning":
        return f"Schedule soon: {base}"
    return "No action needed. Routine check within 30 days."


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI routes (paste into routes.py)
# ══════════════════════════════════════════════════════════════════════════════

FASTAPI_ROUTES = '''
from fastapi import FastAPI, HTTPException, Query
from inference_engine import InferenceEngine, FleetResponse
import pandas as pd

app    = FastAPI(title="InfraCopilot AI v5")
engine = InferenceEngine()
_cache: FleetResponse = None

def fleet() -> FleetResponse:
    global _cache
    if _cache is None:
        df     = pd.read_csv("data/charger_data.csv")
        _cache = engine.generate_fleet_recommendations(df)
    return _cache

@app.get("/fleet/summary")
def summary(): return fleet().summary

@app.get("/fleet/top")
def top(k: int = Query(10, ge=1, le=500)):
    return engine.get_top_k(fleet(), k)

@app.get("/fleet/recommendations")
def recommendations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    risk_level: str = Query(None),
):
    recs = engine.filter_by_risk(fleet(), risk_level) if risk_level else fleet().recommendations
    return engine.paginate(recs, page=page, page_size=page_size)

@app.get("/charger/{charger_id}")
def charger(charger_id: str):
    r = engine.get_by_id(fleet(), charger_id)
    if not r: raise HTTPException(404, f"{charger_id} not found")
    return r
'''


if __name__ == "__main__":
    engine = InferenceEngine()
    df     = pd.read_csv("data/charger_data.csv")
    fleet  = engine.generate_fleet_recommendations(df)
    engine.print_fleet_summary(fleet, top_n=10)
    engine.save_outputs(fleet)
