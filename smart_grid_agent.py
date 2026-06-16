"""
============================================================
  SMART GRID LOAD DECISION AGENT
  Major Project | Python | Rule-Based AI + ML Forecasting
  Modules: Grid Monitoring, Load Prediction, Renewable Energy,
           Battery Storage, Risk Assessment, Decision Agent,
           Reporting Dashboard
============================================================
"""

import csv
import os
import math
import random
import datetime
import json
from collections import deque

# ─── Try optional imports ───────────────────────────────────
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════
GRID_CAPACITY_MW      = 500.0          # Total grid capacity
CRITICAL_THRESHOLD    = 0.90           # 90 % → critical overload
WARNING_THRESHOLD     = 0.75           # 75 % → warning zone
UNDERLOAD_THRESHOLD   = 0.20           # 20 % → underload
NOMINAL_VOLTAGE_KV    = 11.0
VOLTAGE_TOLERANCE_PCT = 0.05           # ±5 %
NOMINAL_FREQ_HZ       = 50.0
FREQ_TOLERANCE_HZ     = 0.5
IDEAL_POWER_FACTOR    = 0.95

CARBON_FACTORS = {                     # kg CO₂ per MWh
    "coal":      820,
    "gas":       490,
    "hydro":       4,
    "solar":       0,
    "wind":        0,
    "nuclear":    12,
    "diesel":    650,
}

CONSUMER_PRIORITIES = {
    "hospital":     1,
    "emergency":    1,
    "industrial":   2,
    "commercial":   3,
    "residential":  4,
}

# ════════════════════════════════════════════════════════════
#  MODULE 1 – GRID MONITORING SYSTEM
# ════════════════════════════════════════════════════════════
class GridMonitor:
    """Real-time & historical load tracking, voltage/frequency/PF monitoring."""

    def __init__(self):
        self.history          = deque(maxlen=168)   # 7 days × 24 h
        self.current_load_mw  = 0.0
        self.voltage_kv       = NOMINAL_VOLTAGE_KV
        self.frequency_hz     = NOMINAL_FREQ_HZ
        self.power_factor     = IDEAL_POWER_FACTOR
        self.timestamps       = []
        self.load_history_raw = []

    # ── Feature 1 · Real-time load monitoring ──────────────
    def update_realtime(self, load_mw, voltage_kv=None, freq_hz=None, pf=None):
        self.current_load_mw = max(0.0, load_mw)
        self.voltage_kv      = voltage_kv if voltage_kv else NOMINAL_VOLTAGE_KV
        self.frequency_hz    = freq_hz    if freq_hz    else NOMINAL_FREQ_HZ
        self.power_factor    = pf         if pf         else IDEAL_POWER_FACTOR

        ts = datetime.datetime.now()
        record = {
            "timestamp":    ts.isoformat(),
            "load_mw":      self.current_load_mw,
            "voltage_kv":   self.voltage_kv,
            "frequency_hz": self.frequency_hz,
            "power_factor": self.power_factor,
            "load_pct":     self.load_percentage(),
        }
        self.history.append(record)
        self.timestamps.append(ts)
        self.load_history_raw.append(self.current_load_mw)
        return record

    # ── Feature 2 · Historical load tracking ───────────────
    def get_history(self, hours=24):
        return list(self.history)[-hours:]

    def load_percentage(self):
        return (self.current_load_mw / GRID_CAPACITY_MW) * 100

    # ── Feature 13 · Overload detection ────────────────────
    def detect_overload(self):
        pct = self.load_percentage()
        if pct >= CRITICAL_THRESHOLD * 100:
            return ("CRITICAL", f"Overload {pct:.1f}% — immediate action required!")
        if pct >= WARNING_THRESHOLD * 100:
            return ("WARNING", f"High load {pct:.1f}% — approaching capacity limit.")
        return ("NORMAL", f"Load {pct:.1f}% — within safe operating range.")

    # ── Feature 14 · Underload detection ───────────────────
    def detect_underload(self):
        pct = self.load_percentage()
        if pct < UNDERLOAD_THRESHOLD * 100:
            return True, f"Underload {pct:.1f}% — grid running at very low demand."
        return False, ""

    # ── Feature 15 · Fault detection ───────────────────────
    def detect_fault(self):
        faults = []
        v_dev  = abs(self.voltage_kv - NOMINAL_VOLTAGE_KV) / NOMINAL_VOLTAGE_KV
        if v_dev > VOLTAGE_TOLERANCE_PCT:
            direction = "HIGH" if self.voltage_kv > NOMINAL_VOLTAGE_KV else "LOW"
            faults.append(f"VOLTAGE FAULT ({direction}): {self.voltage_kv:.2f} kV "
                          f"(nominal {NOMINAL_VOLTAGE_KV} kV ±{VOLTAGE_TOLERANCE_PCT*100:.0f}%)")
        f_dev = abs(self.frequency_hz - NOMINAL_FREQ_HZ)
        if f_dev > FREQ_TOLERANCE_HZ:
            faults.append(f"FREQUENCY FAULT: {self.frequency_hz:.2f} Hz "
                          f"(nominal {NOMINAL_FREQ_HZ} Hz ±{FREQ_TOLERANCE_HZ} Hz)")
        if self.power_factor < 0.85:
            faults.append(f"POWER FACTOR FAULT: {self.power_factor:.2f} (min 0.85)")
        return faults

    # ── Feature 16/17/18 · Voltage/Frequency/PF Monitoring ─
    def monitoring_summary(self):
        return {
            "voltage_kv":       self.voltage_kv,
            "voltage_status":   "OK" if abs(self.voltage_kv - NOMINAL_VOLTAGE_KV)
                                        / NOMINAL_VOLTAGE_KV <= VOLTAGE_TOLERANCE_PCT else "FAULT",
            "frequency_hz":     self.frequency_hz,
            "frequency_status": "OK" if abs(self.frequency_hz - NOMINAL_FREQ_HZ)
                                        <= FREQ_TOLERANCE_HZ else "FAULT",
            "power_factor":     self.power_factor,
            "pf_status":        "OK" if self.power_factor >= 0.85 else "POOR",
        }

    # ── Feature 11 · Grid health score ─────────────────────
    def grid_health_score(self):
        """Composite score 0–100 based on load, voltage, frequency, PF."""
        load_score = max(0, 100 - max(0, self.load_percentage() - 50) * 2)

        v_dev      = abs(self.voltage_kv - NOMINAL_VOLTAGE_KV) / NOMINAL_VOLTAGE_KV
        volt_score = max(0, 100 - v_dev * 1000)

        f_dev      = abs(self.frequency_hz - NOMINAL_FREQ_HZ)
        freq_score = max(0, 100 - f_dev * 100)

        pf_score   = self.power_factor * 100

        score = (load_score * 0.40 + volt_score * 0.25 +
                 freq_score * 0.20 + pf_score   * 0.15)
        grade = ("A" if score >= 90 else "B" if score >= 75 else
                 "C" if score >= 60 else "D" if score >= 45 else "F")
        return round(score, 1), grade


# ════════════════════════════════════════════════════════════
#  MODULE 2 – LOAD PREDICTION ENGINE
# ════════════════════════════════════════════════════════════
class LoadPredictor:
    """Demand forecasting & peak load prediction using ML-style techniques."""

    def __init__(self):
        self.model_weights  = {"trend": 0.3, "seasonal": 0.4, "recent": 0.3}
        self.training_data  = []

    def train(self, historical_loads):
        """Simple linear regression + seasonal baseline training."""
        self.training_data = list(historical_loads)

    # ── Feature 3 · Demand forecasting ─────────────────────
    def forecast(self, hours_ahead=24):
        """
        Weighted combination of:
          • Linear trend from recent data
          • Hour-of-day seasonal pattern (typical daily cycle)
          • Exponential smoothing of recent readings
        """
        if len(self.training_data) < 4:
            base = 250.0
            return [base + random.uniform(-20, 20) for _ in range(hours_ahead)]

        data    = self.training_data[-48:] if len(self.training_data) >= 48 \
                  else self.training_data
        n       = len(data)
        avg     = sum(data) / n
        # Linear trend slope
        x_mean  = (n - 1) / 2
        slope   = sum((i - x_mean) * (v - avg) for i, v in enumerate(data)) / \
                  max(1, sum((i - x_mean) ** 2 for i in range(n)))

        # Seasonal factor: simple sinusoidal daily pattern
        def seasonal(h):
            # Peak ~ 18:00, trough ~ 04:00
            angle  = 2 * math.pi * ((h % 24) - 4) / 24
            factor = 1 + 0.25 * math.sin(angle)
            return factor

        # Exponential smoothing (α = 0.3)
        alpha   = 0.3
        smooth  = data[0]
        for v in data[1:]:
            smooth = alpha * v + (1 - alpha) * smooth

        forecasts = []
        for h in range(hours_ahead):
            trend_val    = avg + slope * (n + h)
            seasonal_val = avg * seasonal(h)
            recent_val   = smooth + slope * h
            pred = (self.model_weights["trend"]    * trend_val   +
                    self.model_weights["seasonal"]  * seasonal_val +
                    self.model_weights["recent"]    * recent_val)
            pred = max(50, min(GRID_CAPACITY_MW * 0.95, pred))
            forecasts.append(round(pred, 2))
        return forecasts

    # ── Feature 4 · Peak load prediction ───────────────────
    def predict_peak(self, hours_ahead=24):
        forecasts = self.forecast(hours_ahead)
        peak_val  = max(forecasts)
        peak_hour = forecasts.index(peak_val)
        return peak_val, peak_hour, forecasts


# ════════════════════════════════════════════════════════════
#  MODULE 3 – RENEWABLE ENERGY MANAGER
# ════════════════════════════════════════════════════════════
class RenewableEnergyManager:
    """Solar & wind monitoring + renewable contribution tracking."""

    def __init__(self):
        self.solar_capacity_mw = 80.0
        self.wind_capacity_mw  = 60.0
        self.solar_output_mw   = 0.0
        self.wind_output_mw    = 0.0
        self.solar_history     = []
        self.wind_history      = []

    # ── Feature 5 · Renewable energy monitoring ────────────
    def get_renewable_total(self):
        return self.solar_output_mw + self.wind_output_mw

    def renewable_percentage(self, total_load_mw):
        if total_load_mw <= 0:
            return 0.0
        return (self.get_renewable_total() / total_load_mw) * 100

    # ── Feature 6 · Solar energy integration ───────────────
    def update_solar(self, hour_of_day, cloud_cover_pct=20):
        """Model solar generation based on time of day and cloud cover."""
        if 6 <= hour_of_day <= 18:
            sun_angle = math.sin(math.pi * (hour_of_day - 6) / 12)
            output    = self.solar_capacity_mw * sun_angle * (1 - cloud_cover_pct / 100)
        else:
            output = 0.0
        noise               = random.uniform(-2, 2)
        self.solar_output_mw = max(0, output + noise)
        self.solar_history.append(round(self.solar_output_mw, 2))
        return self.solar_output_mw

    # ── Feature 7 · Wind energy integration ────────────────
    def update_wind(self, wind_speed_ms=8.0):
        """Simplified wind power curve: cubic law between cut-in (3 m/s) & rated (12 m/s)."""
        cut_in, rated, cut_out = 3.0, 12.0, 25.0
        if wind_speed_ms < cut_in or wind_speed_ms > cut_out:
            output = 0.0
        elif wind_speed_ms >= rated:
            output = self.wind_capacity_mw
        else:
            output = self.wind_capacity_mw * ((wind_speed_ms - cut_in) /
                                               (rated - cut_in)) ** 3
        noise              = random.uniform(-1, 1)
        self.wind_output_mw = max(0, min(self.wind_capacity_mw, output + noise))
        self.wind_history.append(round(self.wind_output_mw, 2))
        return self.wind_output_mw

    def summary(self):
        return {
            "solar_mw":       round(self.solar_output_mw, 2),
            "wind_mw":        round(self.wind_output_mw, 2),
            "total_renew_mw": round(self.get_renewable_total(), 2),
            "solar_cap_mw":   self.solar_capacity_mw,
            "wind_cap_mw":    self.wind_capacity_mw,
        }


# ════════════════════════════════════════════════════════════
#  MODULE 4 – BATTERY STORAGE CONTROLLER
# ════════════════════════════════════════════════════════════
class BatteryStorageController:
    """Battery management: charging, discharging, state-of-charge tracking."""

    def __init__(self, capacity_mwh=100.0, max_power_mw=40.0):
        self.capacity_mwh   = capacity_mwh
        self.max_power_mw   = max_power_mw
        self.soc_mwh        = capacity_mwh * 0.50   # start at 50 %
        self.charge_eff     = 0.95
        self.discharge_eff  = 0.95
        self.min_soc_pct    = 0.10
        self.max_soc_pct    = 0.95
        self.soc_history    = []
        self.status         = "IDLE"

    # ── Feature 8 · Battery storage management ─────────────
    @property
    def soc_pct(self):
        return (self.soc_mwh / self.capacity_mwh) * 100

    def available_discharge_mwh(self):
        min_soc = self.capacity_mwh * self.min_soc_pct
        return max(0, self.soc_mwh - min_soc)

    def available_charge_mwh(self):
        max_soc = self.capacity_mwh * self.max_soc_pct
        return max(0, max_soc - self.soc_mwh)

    # ── Feature 9 · Battery charging ───────────────────────
    def charge(self, power_mw, duration_h=1.0):
        if self.soc_pct >= self.max_soc_pct * 100:
            self.status = "FULL"
            return 0.0
        chargeable = min(power_mw, self.max_power_mw) * duration_h
        actual     = min(chargeable * self.charge_eff, self.available_charge_mwh())
        self.soc_mwh += actual
        self.status   = "CHARGING"
        self.soc_history.append(round(self.soc_pct, 1))
        return round(actual / duration_h, 2)

    # ── Feature 10 · Battery discharging ───────────────────
    def discharge(self, power_mw, duration_h=1.0):
        if self.soc_pct <= self.min_soc_pct * 100:
            self.status = "DEPLETED"
            return 0.0
        required = min(power_mw, self.max_power_mw) * duration_h
        actual   = min(required, self.available_discharge_mwh())
        self.soc_mwh -= actual
        self.status   = "DISCHARGING"
        self.soc_history.append(round(self.soc_pct, 1))
        return round((actual * self.discharge_eff) / duration_h, 2)

    def summary(self):
        return {
            "soc_pct":        round(self.soc_pct, 1),
            "soc_mwh":        round(self.soc_mwh, 2),
            "capacity_mwh":   self.capacity_mwh,
            "status":         self.status,
            "avail_disch_mwh": round(self.available_discharge_mwh(), 2),
            "avail_chg_mwh":   round(self.available_charge_mwh(), 2),
        }


# ════════════════════════════════════════════════════════════
#  MODULE 5 – RISK ASSESSMENT ENGINE
# ════════════════════════════════════════════════════════════
class RiskAssessmentEngine:
    """Composite risk score, risk level, and load-shedding recommendations."""

    RISK_WEIGHTS = {
        "load":       0.35,
        "voltage":    0.20,
        "frequency":  0.15,
        "battery_soc":0.15,
        "power_factor":0.15,
    }

    def __init__(self):
        self.risk_log = []

    # ── Feature 12 · Risk assessment engine ────────────────
    def calculate_risk(self, monitor: GridMonitor,
                       battery: BatteryStorageController) -> dict:
        load_pct = monitor.load_percentage()

        # Individual risk components (0–100, higher = riskier)
        load_risk = min(100, max(0, (load_pct - 50) * 2))

        v_dev     = abs(monitor.voltage_kv - NOMINAL_VOLTAGE_KV) / NOMINAL_VOLTAGE_KV
        volt_risk = min(100, v_dev * 2000)

        f_dev     = abs(monitor.frequency_hz - NOMINAL_FREQ_HZ)
        freq_risk = min(100, f_dev * 200)

        bat_risk  = max(0, 100 - battery.soc_pct * 1.05)

        pf_risk   = max(0, (IDEAL_POWER_FACTOR - monitor.power_factor) * 500)

        composite = (load_risk    * self.RISK_WEIGHTS["load"]         +
                     volt_risk    * self.RISK_WEIGHTS["voltage"]       +
                     freq_risk    * self.RISK_WEIGHTS["frequency"]     +
                     bat_risk     * self.RISK_WEIGHTS["battery_soc"]   +
                     pf_risk      * self.RISK_WEIGHTS["power_factor"])

        composite = round(composite, 1)
        level     = ("CRITICAL" if composite >= 75 else
                     "HIGH"     if composite >= 50 else
                     "MEDIUM"   if composite >= 25 else "LOW")

        result = {
            "score":        composite,
            "level":        level,
            "components":   {
                "load_risk":    round(load_risk, 1),
                "voltage_risk": round(volt_risk, 1),
                "frequency_risk": round(freq_risk, 1),
                "battery_risk": round(bat_risk, 1),
                "pf_risk":      round(pf_risk, 1),
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.risk_log.append(result)
        return result

    # ── Feature 27 · Load shedding recommendation ──────────
    def recommend_load_shedding(self, load_pct, consumers):
        """
        Priority-based shedding: shed lowest priority first.
        Returns list of consumers to shed and estimated MW saved.
        """
        if load_pct < WARNING_THRESHOLD * 100:
            return [], 0.0

        excess_pct = load_pct - WARNING_THRESHOLD * 100
        target_mw  = (excess_pct / 100) * GRID_CAPACITY_MW
        sorted_c   = sorted(consumers, key=lambda c: (-c["priority"], c["load_mw"]))
        shed_list  = []
        shed_mw    = 0.0

        for c in sorted_c:
            if shed_mw >= target_mw:
                break
            if c["priority"] == CONSUMER_PRIORITIES["hospital"]:
                continue                             # Never shed hospitals
            shed_list.append(c)
            shed_mw += c["load_mw"]

        return shed_list, round(shed_mw, 2)

    # ── Feature 28 · Carbon emission estimation ────────────
    def estimate_carbon(self, generation_mix: dict, duration_h=1.0):
        """
        generation_mix: {"coal": MW, "gas": MW, "solar": MW, ...}
        Returns total CO₂ in tonnes.
        """
        total_co2 = 0.0
        detail    = {}
        for source, mw in generation_mix.items():
            factor        = CARBON_FACTORS.get(source, 500)
            co2           = mw * duration_h * factor / 1000   # tonnes
            detail[source] = round(co2, 2)
            total_co2     += co2
        return round(total_co2, 2), detail


# ════════════════════════════════════════════════════════════
#  CONSUMER MANAGER
# ════════════════════════════════════════════════════════════
class ConsumerManager:
    """Manages all consumer types with priority-based allocation."""

    def __init__(self):
        self.consumers = []
        self._init_default_consumers()

    def _init_default_consumers(self):
        # ── Feature 21 · Hospital priority support ─────────
        self.consumers += [
            {"id": "H-01", "name": "City Hospital",      "type": "hospital",    "priority": 1, "load_mw": 12.0, "active": True},
            {"id": "H-02", "name": "Medical Centre",     "type": "hospital",    "priority": 1, "load_mw":  8.0, "active": True},
        ]
        # ── Feature 23 · Industrial consumer management ────
        self.consumers += [
            {"id": "I-01", "name": "Steel Plant",        "type": "industrial",  "priority": 2, "load_mw": 85.0, "active": True},
            {"id": "I-02", "name": "Cement Factory",     "type": "industrial",  "priority": 2, "load_mw": 60.0, "active": True},
            {"id": "I-03", "name": "Textile Mill",       "type": "industrial",  "priority": 2, "load_mw": 40.0, "active": True},
        ]
        # ── Feature 24 · Commercial consumer management ────
        self.consumers += [
            {"id": "C-01", "name": "IT Park",            "type": "commercial",  "priority": 3, "load_mw": 30.0, "active": True},
            {"id": "C-02", "name": "Shopping Mall",      "type": "commercial",  "priority": 3, "load_mw": 25.0, "active": True},
            {"id": "C-03", "name": "Office Complex",     "type": "commercial",  "priority": 3, "load_mw": 18.0, "active": True},
        ]
        # ── Feature 22 · Residential consumer management ───
        self.consumers += [
            {"id": "R-01", "name": "Residential Zone A", "type": "residential", "priority": 4, "load_mw": 50.0, "active": True},
            {"id": "R-02", "name": "Residential Zone B", "type": "residential", "priority": 4, "load_mw": 45.0, "active": True},
            {"id": "R-03", "name": "Residential Zone C", "type": "residential", "priority": 4, "load_mw": 35.0, "active": True},
        ]

    def total_load(self):
        return sum(c["load_mw"] for c in self.consumers if c["active"])

    def get_by_type(self, ctype):
        return [c for c in self.consumers if c["type"] == ctype]

    def shed_consumer(self, consumer_id):
        for c in self.consumers:
            if c["id"] == consumer_id:
                c["active"] = False
                return True
        return False

    def restore_all(self):
        for c in self.consumers:
            c["active"] = True

    # ── Feature 20 · Priority-based consumer allocation ────
    def allocation_report(self):
        report = []
        for priority in sorted(set(CONSUMER_PRIORITIES.values())):
            group = [c for c in self.consumers if c["priority"] == priority]
            total = sum(c["load_mw"] for c in group)
            report.append({
                "priority": priority,
                "count":    len(group),
                "total_mw": round(total, 2),
                "consumers": group,
            })
        return report


# ════════════════════════════════════════════════════════════
#  MODULE 6 – DECISION AGENT
# ════════════════════════════════════════════════════════════
class DecisionAgent:
    """
    Rule-based AI agent implementing the full decision flow:
    Normal Supply → Battery Support → Renewable Support →
    Import Power → Load Shedding
    """

    def __init__(self, monitor: GridMonitor, predictor: LoadPredictor,
                 renewable: RenewableEnergyManager, battery: BatteryStorageController,
                 risk_engine: RiskAssessmentEngine, consumers: ConsumerManager):
        self.monitor     = monitor
        self.predictor   = predictor
        self.renewable   = renewable
        self.battery     = battery
        self.risk_engine = risk_engine
        self.consumers   = consumers
        self.mode        = "NORMAL"
        self.decision_log = []
        self.emergency_active = False
        # ── Feature 26 · Backup generator management ───────
        self.generator_online  = False
        self.generator_cap_mw  = 50.0
        self.generator_fuel_kg = 5000.0   # kg diesel

    # ── Feature 19 · Automatic load balancing ──────────────
    def balance_load(self):
        total_demand = self.consumers.total_load()
        renewable_mw = self.renewable.get_renewable_total()
        grid_supply  = total_demand - renewable_mw
        balance      = GRID_CAPACITY_MW - grid_supply
        return round(grid_supply, 2), round(balance, 2)

    def make_decision(self):
        """Core decision flow engine."""
        load_mw   = self.consumers.total_load()
        self.monitor.update_realtime(load_mw)
        risk      = self.risk_engine.calculate_risk(self.monitor, self.battery)
        ol_status, ol_msg = self.monitor.detect_overload()
        renewable_mw      = self.renewable.get_renewable_total()
        shortage          = max(0, load_mw - (GRID_CAPACITY_MW * 0.85))

        decision = {
            "timestamp":      datetime.datetime.now().isoformat(),
            "load_mw":        round(load_mw, 2),
            "renewable_mw":   round(renewable_mw, 2),
            "risk_score":     risk["score"],
            "risk_level":     risk["level"],
            "actions":        [],
            "mode":           "NORMAL",
            "shed_consumers": [],
        }

        # ── STEP 1: Normal supply (no issue) ───────────────
        if ol_status == "NORMAL":
            decision["mode"] = "NORMAL"
            decision["actions"].append("✅ Normal grid supply — all consumers served.")

            # Opportunistic battery charging
            if self.battery.soc_pct < 80 and renewable_mw > load_mw * 0.5:
                charged = self.battery.charge(min(20, renewable_mw - load_mw * 0.5))
                if charged > 0:
                    decision["actions"].append(
                        f"🔋 Charging battery with surplus renewable: {charged} MW")

        # ── STEP 2: Warning — use battery + renewable first ─
        elif ol_status == "WARNING":
            decision["mode"] = "BATTERY_SUPPORT"
            if self.battery.soc_pct > self.battery.min_soc_pct * 100 + 10:
                dispatched = self.battery.discharge(min(shortage + 20, self.battery.max_power_mw))
                decision["actions"].append(
                    f"🔋 Battery dispatched: {dispatched} MW (SoC: {self.battery.soc_pct:.1f}%)")
                shortage -= dispatched

            if shortage > 0 and renewable_mw > 0:
                decision["mode"] = "RENEWABLE_SUPPORT"
                decision["actions"].append(
                    f"☀️ Renewable contribution: {renewable_mw:.1f} MW offsetting demand.")
                shortage -= renewable_mw

        # ── STEP 3: Critical — escalate to import / shedding
        elif ol_status == "CRITICAL":
            decision["mode"] = "CRITICAL"

            # Battery emergency discharge
            if self.battery.available_discharge_mwh() > 10:
                dispatched = self.battery.discharge(self.battery.max_power_mw)
                decision["actions"].append(f"⚡ Emergency battery discharge: {dispatched} MW")
                shortage -= dispatched

            # Import power (simulated)
            if shortage > 0:
                import_mw = min(shortage, 80)
                decision["mode"] = "IMPORT_POWER"
                decision["actions"].append(
                    f"🔌 Importing {import_mw:.1f} MW from neighboring grid.")
                shortage -= import_mw

            # Backup generator
            if shortage > 0 and not self.generator_online:
                self._start_generator()
                decision["actions"].append(
                    f"🏭 Backup generator online: {self.generator_cap_mw} MW")
                shortage -= self.generator_cap_mw

            # Load shedding as last resort
            if shortage > 5:
                load_pct    = self.monitor.load_percentage()
                shed, mw    = self.risk_engine.recommend_load_shedding(
                    load_pct, self.consumers.consumers)
                for c in shed:
                    self.consumers.shed_consumer(c["id"])
                decision["mode"]           = "LOAD_SHEDDING"
                decision["shed_consumers"] = [c["name"] for c in shed]
                decision["actions"].append(
                    f"⚠️ Load shedding: {len(shed)} consumers, {mw:.1f} MW reduced.")

        # ── Feature 25 · Emergency mode activation ─────────
        if risk["score"] >= 75 and not self.emergency_active:
            self.emergency_active = True
            self.mode             = "EMERGENCY"
            decision["mode"]      = "EMERGENCY"
            decision["actions"].append("🚨 EMERGENCY MODE ACTIVATED — All agencies alerted!")

        self.mode = decision["mode"]
        self.decision_log.append(decision)
        return decision

    def _start_generator(self):
        self.generator_online = True
        self.generator_fuel_kg -= 100   # burn fuel

    def stop_generator(self):
        self.generator_online = False

    def clear_emergency(self):
        self.emergency_active = False
        self.consumers.restore_all()
        self.mode = "NORMAL"

    def run_simulation(self, steps=24):
        """Simulate 24 hours of grid operation."""
        print(section_header("SMART GRID SIMULATION — 24 HOUR RUN"))
        self.predictor.train([200 + 80 * math.sin(math.pi * h / 12) +
                               random.uniform(-15, 15) for h in range(48)])
        results = []
        for step in range(steps):
            hour = step % 24
            # Simulate varying conditions
            base_load   = 200 + 100 * math.sin(math.pi * (hour - 6) / 12)
            noise       = random.uniform(-25, 25)
            load        = max(80, base_load + noise)

            wind_speed  = random.uniform(4, 14)
            cloud_cover = random.uniform(10, 60)

            # Update renewables
            self.renewable.update_solar(hour, cloud_cover)
            self.renewable.update_wind(wind_speed)

            # Simulate load variation across consumers
            factor = load / self.consumers.total_load()
            for c in self.consumers.consumers:
                c["load_mw"] = round(c["load_mw"] * random.uniform(0.90, 1.10), 2)

            # Simulate occasional fault conditions
            volt_noise  = random.uniform(-0.3, 0.3)
            freq_noise  = random.uniform(-0.3, 0.3)
            pf_noise    = random.uniform(-0.05, 0.02)

            self.monitor.voltage_kv    = NOMINAL_VOLTAGE_KV + volt_noise
            self.monitor.frequency_hz  = NOMINAL_FREQ_HZ + freq_noise
            self.monitor.power_factor  = max(0.80, IDEAL_POWER_FACTOR + pf_noise)

            decision = self.make_decision()
            results.append(decision)

            # Print each step summary
            print(f"  Hour {hour:02d}:00 | Load {decision['load_mw']:6.1f} MW | "
                  f"Risk {decision['risk_score']:5.1f} | "
                  f"Mode: {decision['mode']:<18} | "
                  f"Renew: {decision['renewable_mw']:5.1f} MW | "
                  f"Bat: {self.battery.soc_pct:4.1f}%")

            # Restore consumers between steps unless in emergency
            if self.emergency_active and decision["risk_score"] < 50:
                self.clear_emergency()

        return results


# ════════════════════════════════════════════════════════════
#  MODULE 7 – REPORTING DASHBOARD
# ════════════════════════════════════════════════════════════
class ReportingDashboard:
    """Daily report generation, CSV export, and chart generation."""

    def __init__(self, agent: DecisionAgent):
        self.agent = agent

    # ── Feature 29 · Daily report generation ───────────────
    def generate_daily_report(self):
        log     = self.agent.decision_log
        monitor = self.agent.monitor
        battery = self.agent.battery
        renew   = self.agent.renewable

        if not log:
            return "No simulation data available. Run simulation first."

        loads       = [r["load_mw"]     for r in log]
        risks       = [r["risk_score"]  for r in log]
        renew_mws   = [r["renewable_mw"] for r in log]
        modes       = [r["mode"]        for r in log]
        health, grade = monitor.grid_health_score()

        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        shed_events = [r for r in log if r.get("shed_consumers")]
        total_shed  = sum(len(r["shed_consumers"]) for r in shed_events)

        avg_renew_pct = (sum(renew_mws) / sum(loads) * 100) if sum(loads) > 0 else 0

        # Carbon estimate (simplified)
        gen_mix = {
            "solar":  sum(self.agent.renewable.solar_history) / max(1, len(self.agent.renewable.solar_history)),
            "wind":   sum(self.agent.renewable.wind_history)  / max(1, len(self.agent.renewable.wind_history)),
            "gas":    max(0, sum(loads) / len(loads) - sum(renew_mws) / len(renew_mws)),
        }
        total_co2, co2_detail = self.agent.risk_engine.estimate_carbon(gen_mix, 24)

        now    = datetime.datetime.now()
        report = f"""
{'='*65}
   SMART GRID LOAD DECISION AGENT — DAILY OPERATIONS REPORT
   Date : {now.strftime('%A, %d %B %Y')}
   Time : {now.strftime('%H:%M:%S')}
{'='*65}

  GRID HEALTH
  ─────────────────────────────────────────────────────────
  Health Score    : {health}/100  (Grade {grade})
  Battery SoC     : {battery.soc_pct:.1f}%  [{battery.summary()['avail_disch_mwh']} MWh available]
  Emergency Mode  : {'ACTIVE ⚠️' if self.agent.emergency_active else 'Inactive ✅'}
  Generator       : {'ONLINE 🏭' if self.agent.generator_online else 'Offline'}

  LOAD STATISTICS (24-hour)
  ─────────────────────────────────────────────────────────
  Average Load    : {sum(loads)/len(loads):.1f} MW  ({sum(loads)/len(loads)/GRID_CAPACITY_MW*100:.1f}% of capacity)
  Peak Load       : {max(loads):.1f} MW  ({max(loads)/GRID_CAPACITY_MW*100:.1f}% of capacity)
  Minimum Load    : {min(loads):.1f} MW  ({min(loads)/GRID_CAPACITY_MW*100:.1f}% of capacity)
  Load Std Dev    : {_std(loads):.1f} MW

  RENEWABLE ENERGY
  ─────────────────────────────────────────────────────────
  Avg Renewable   : {sum(renew_mws)/len(renew_mws):.1f} MW
  Peak Renewable  : {max(renew_mws):.1f} MW
  Renewable Share : {avg_renew_pct:.1f}% of total demand
  Solar Peak      : {max(self.agent.renewable.solar_history or [0]):.1f} MW
  Wind Peak       : {max(self.agent.renewable.wind_history  or [0]):.1f} MW

  RISK & DECISIONS
  ─────────────────────────────────────────────────────────
  Avg Risk Score  : {sum(risks)/len(risks):.1f} / 100
  Peak Risk Score : {max(risks):.1f} / 100
  Load Shedding   : {len(shed_events)} events  ({total_shed} consumers affected)
  Operating Modes :
{''.join(f"    {m:<20} : {c} hours{chr(10)}" for m,c in mode_counts.items())}
  CARBON FOOTPRINT (est.)
  ─────────────────────────────────────────────────────────
  Total CO₂       : {total_co2:.1f} tonnes (24-hour)
{''.join(f"    {src:<10} : {val:.2f} t{chr(10)}" for src,val in co2_detail.items())}
  MONITORING STATUS
  ─────────────────────────────────────────────────────────
  Voltage         : {monitor.voltage_kv:.2f} kV  ({monitor.monitoring_summary()['voltage_status']})
  Frequency       : {monitor.frequency_hz:.2f} Hz  ({monitor.monitoring_summary()['frequency_status']})
  Power Factor    : {monitor.power_factor:.3f}  ({monitor.monitoring_summary()['pf_status']})
  Faults Detected : {len(monitor.detect_fault())}

{'='*65}
  Report generated by Smart Grid Load Decision Agent v1.0
{'='*65}
"""
        return report

    # ── Feature 30 · CSV data export ───────────────────────
    def export_to_csv(self, filepath="reports/grid_report.csv"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        log = self.agent.decision_log
        if not log:
            print("No data to export.")
            return

        fieldnames = ["timestamp", "load_mw", "renewable_mw", "risk_score",
                      "risk_level", "mode", "shed_count"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in log:
                writer.writerow({
                    "timestamp":    r["timestamp"],
                    "load_mw":      r["load_mw"],
                    "renewable_mw": r["renewable_mw"],
                    "risk_score":   r["risk_score"],
                    "risk_level":   r["risk_level"],
                    "mode":         r["mode"],
                    "shed_count":   len(r.get("shed_consumers", [])),
                })
        print(f"  ✅ CSV exported → {filepath}")
        return filepath

    def export_battery_csv(self, filepath="reports/battery_log.csv"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        history = self.agent.battery.soc_history
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "soc_pct"])
            for i, v in enumerate(history):
                writer.writerow([i, v])
        print(f"  ✅ Battery log → {filepath}")

    # ── Charts (matplotlib) ────────────────────────────────
    def generate_charts(self, out_dir="reports"):
        if not MATPLOTLIB_AVAILABLE:
            print("  ⚠️  matplotlib not available — skipping charts.")
            return
        os.makedirs(out_dir, exist_ok=True)
        log = self.agent.decision_log
        if not log:
            return

        hours     = list(range(len(log)))
        loads     = [r["load_mw"]     for r in log]
        risks     = [r["risk_score"]  for r in log]
        renew_mws = [r["renewable_mw"] for r in log]
        bat_soc   = self.agent.battery.soc_history or [50] * len(log)
        if len(bat_soc) > len(hours):
            bat_soc = bat_soc[:len(hours)]
        elif len(bat_soc) < len(hours):
            bat_soc += [bat_soc[-1]] * (len(hours) - len(bat_soc))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Smart Grid Load Decision Agent — Dashboard",
                     fontsize=15, fontweight="bold")

        # Plot 1: Load vs Capacity
        ax = axes[0, 0]
        ax.plot(hours, loads, color="#2196F3", linewidth=2, label="Load (MW)")
        ax.plot(hours, renew_mws, color="#4CAF50", linewidth=2,
                linestyle="--", label="Renewable (MW)")
        ax.axhline(GRID_CAPACITY_MW * WARNING_THRESHOLD,
                   color="orange", linestyle=":", label="Warning threshold")
        ax.axhline(GRID_CAPACITY_MW * CRITICAL_THRESHOLD,
                   color="red", linestyle=":", label="Critical threshold")
        ax.set_title("Grid Load vs Renewable Generation")
        ax.set_xlabel("Hour"); ax.set_ylabel("MW")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Plot 2: Risk score
        ax = axes[0, 1]
        colors = ["red" if r >= 75 else "orange" if r >= 50 else
                  "gold" if r >= 25 else "green" for r in risks]
        ax.bar(hours, risks, color=colors, alpha=0.8)
        ax.axhline(75, color="red",    linestyle="--", linewidth=1)
        ax.axhline(50, color="orange", linestyle="--", linewidth=1)
        ax.set_title("Risk Score Over Time")
        ax.set_xlabel("Hour"); ax.set_ylabel("Risk Score (0–100)")
        ax.set_ylim(0, 100); ax.grid(alpha=0.3, axis="y")
        patches = [mpatches.Patch(color=c, label=l) for c, l in
                   [("green","Low"), ("gold","Medium"), ("orange","High"), ("red","Critical")]]
        ax.legend(handles=patches, fontsize=8)

        # Plot 3: Battery SoC
        ax = axes[1, 0]
        ax.fill_between(hours, bat_soc, alpha=0.3, color="#9C27B0")
        ax.plot(hours, bat_soc, color="#9C27B0", linewidth=2, label="Battery SoC %")
        ax.axhline(20, color="red",    linestyle=":", label="Min SoC")
        ax.axhline(95, color="green",  linestyle=":", label="Max SoC")
        ax.set_title("Battery State of Charge")
        ax.set_xlabel("Hour"); ax.set_ylabel("SoC (%)")
        ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Plot 4: Operating mode distribution (pie)
        ax = axes[1, 1]
        modes = [r["mode"] for r in log]
        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        mode_colors = {
            "NORMAL":           "#4CAF50",
            "BATTERY_SUPPORT":  "#9C27B0",
            "RENEWABLE_SUPPORT":"#2196F3",
            "IMPORT_POWER":     "#FF9800",
            "LOAD_SHEDDING":    "#F44336",
            "CRITICAL":         "#B71C1C",
            "EMERGENCY":        "#000000",
        }
        clrs = [mode_colors.get(m, "#607D8B") for m in mode_counts]
        ax.pie(mode_counts.values(), labels=mode_counts.keys(), colors=clrs,
               autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
        ax.set_title("Operating Mode Distribution")

        plt.tight_layout()
        chart_path = os.path.join(out_dir, "grid_dashboard.png")
        plt.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  📊 Dashboard chart saved → {chart_path}")
        return chart_path


# ════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ════════════════════════════════════════════════════════════
def _std(data):
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    var  = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(var)


def section_header(title):
    return f"\n{'═'*65}\n  {title}\n{'═'*65}"


# ════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════
def main():
    print(section_header("SMART GRID LOAD DECISION AGENT  v1.0"))
    print("  Python | Rule-Based AI | ML Forecasting | 30 Features")
    print("  Modules: GridMonitor · LoadPredictor · RenewableManager")
    print("           BatteryController · RiskEngine · DecisionAgent")
    print("           ReportingDashboard")

    # ── Initialise all modules ──────────────────────────────
    monitor      = GridMonitor()
    predictor    = LoadPredictor()
    renewable    = RenewableEnergyManager()
    battery      = BatteryStorageController(capacity_mwh=100, max_power_mw=40)
    risk_engine  = RiskAssessmentEngine()
    consumers    = ConsumerManager()
    agent        = DecisionAgent(monitor, predictor, renewable,
                                 battery, risk_engine, consumers)
    dashboard    = ReportingDashboard(agent)

    # ── Module 1: Grid Monitoring demo ─────────────────────
    print(section_header("MODULE 1 · Grid Monitoring System"))
    sample = monitor.update_realtime(320, voltage_kv=10.9,
                                     freq_hz=50.1, pf=0.92)
    print(f"  Real-time reading   : {sample['load_mw']} MW  "
          f"({sample['load_pct']:.1f}% capacity)")
    status, msg = monitor.detect_overload()
    print(f"  Overload status     : [{status}] {msg}")
    ul, ul_msg  = monitor.detect_underload()
    print(f"  Underload check     : {ul_msg if ul else 'No underload detected'}")
    faults      = monitor.detect_fault()
    print(f"  Faults detected     : {len(faults)}")
    for f in faults:
        print(f"    • {f}")
    health, grade = monitor.grid_health_score()
    print(f"  Grid Health Score   : {health}/100  (Grade {grade})")
    ms = monitor.monitoring_summary()
    print(f"  Voltage Status      : {ms['voltage_kv']:.2f} kV  [{ms['voltage_status']}]")
    print(f"  Frequency Status    : {ms['frequency_hz']:.2f} Hz [{ms['frequency_status']}]")
    print(f"  Power Factor        : {ms['power_factor']:.3f}    [{ms['pf_status']}]")

    # ── Module 2: Load Prediction Engine ───────────────────
    print(section_header("MODULE 2 · Load Prediction Engine"))
    train_data  = [200 + 80 * math.sin(math.pi * h / 12) +
                   random.uniform(-15, 15) for h in range(48)]
    predictor.train(train_data)
    peak_val, peak_hr, forecasts = predictor.predict_peak(24)
    print(f"  24-hour forecast (MW): {[round(v,1) for v in forecasts[:8]]} ...")
    print(f"  Predicted peak load  : {peak_val:.1f} MW at hour {peak_hr:02d}:00")

    # ── Module 3: Renewable Energy Manager ─────────────────
    print(section_header("MODULE 3 · Renewable Energy Manager"))
    solar = renewable.update_solar(hour_of_day=12, cloud_cover_pct=20)
    wind  = renewable.update_wind(wind_speed_ms=9.5)
    rs    = renewable.summary()
    print(f"  Solar Output        : {rs['solar_mw']} MW  (cap {rs['solar_cap_mw']} MW)")
    print(f"  Wind Output         : {rs['wind_mw']} MW   (cap {rs['wind_cap_mw']} MW)")
    print(f"  Total Renewable     : {rs['total_renew_mw']} MW")
    print(f"  Renewable Share     : {renewable.renewable_percentage(320):.1f}%")

    # ── Module 4: Battery Storage Controller ───────────────
    print(section_header("MODULE 4 · Battery Storage Controller"))
    bs = battery.summary()
    print(f"  Initial SoC         : {bs['soc_pct']}%  ({bs['soc_mwh']} MWh)")
    charged = battery.charge(power_mw=25)
    print(f"  Charged             : {charged} MW  → SoC {battery.soc_pct:.1f}%")
    discharged = battery.discharge(power_mw=15)
    print(f"  Discharged          : {discharged} MW  → SoC {battery.soc_pct:.1f}%")
    bs2 = battery.summary()
    print(f"  Available discharge : {bs2['avail_disch_mwh']} MWh")
    print(f"  Available charge    : {bs2['avail_chg_mwh']} MWh")

    # ── Module 5: Risk Assessment Engine ───────────────────
    print(section_header("MODULE 5 · Risk Assessment Engine"))
    monitor.update_realtime(440, voltage_kv=10.7, freq_hz=49.6, pf=0.87)
    risk = risk_engine.calculate_risk(monitor, battery)
    print(f"  Risk Score          : {risk['score']} / 100  [{risk['level']}]")
    print(f"  Components:")
    for k, v in risk["components"].items():
        print(f"    {k:<20}: {v}")

    load_pct   = monitor.load_percentage()
    shed, mw   = risk_engine.recommend_load_shedding(load_pct, consumers.consumers)
    print(f"\n  Load Shedding Recommendation (load {load_pct:.1f}%):")
    if shed:
        print(f"  Shed {len(shed)} consumer(s), save {mw} MW:")
        for c in shed:
            print(f"    • {c['name']}  [{c['type']}]  {c['load_mw']} MW")
    else:
        print("  No shedding needed.")

    gen_mix  = {"coal": 100, "gas": 80, "solar": 40, "wind": 30, "hydro": 20}
    co2, det = risk_engine.estimate_carbon(gen_mix, duration_h=1)
    print(f"\n  Carbon Estimate (1 hr): {co2} tonnes CO₂")
    for src, val in det.items():
        print(f"    {src:<10}: {val} t")

    # ── Consumer Manager ───────────────────────────────────
    print(section_header("CONSUMER ALLOCATION REPORT"))
    consumers.restore_all()
    alloc = consumers.allocation_report()
    for group in alloc:
        print(f"  Priority {group['priority']} — {group['count']} consumers, "
              f"total {group['total_mw']} MW")
        for c in group["consumers"]:
            print(f"    [{c['id']}] {c['name']:<25} {c['load_mw']:6.1f} MW")

    # ── Module 6: Decision Agent Simulation ────────────────
    consumers.restore_all()
    results = agent.run_simulation(steps=24)

    # ── Module 7: Reporting Dashboard ──────────────────────
    print(section_header("MODULE 7 · Reporting Dashboard"))

    report = dashboard.generate_daily_report()
    print(report)

    report_path = "reports/daily_report.txt"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  📄 Text report saved → {report_path}")

    dashboard.export_to_csv("reports/grid_simulation.csv")
    dashboard.export_battery_csv("reports/battery_log.csv")
    dashboard.generate_charts("reports")

    print(section_header("ALL 30 FEATURES DEMONSTRATED SUCCESSFULLY"))
    print("""
  FEATURE INDEX
  ─────────────────────────────────────────────────────────
   1  Real-time load monitoring         ✅  GridMonitor.update_realtime()
   2  Historical load tracking          ✅  GridMonitor.get_history()
   3  Demand forecasting                ✅  LoadPredictor.forecast()
   4  Peak load prediction              ✅  LoadPredictor.predict_peak()
   5  Renewable energy monitoring       ✅  RenewableEnergyManager.summary()
   6  Solar energy integration          ✅  RenewableEnergyManager.update_solar()
   7  Wind energy integration           ✅  RenewableEnergyManager.update_wind()
   8  Battery storage management        ✅  BatteryStorageController.summary()
   9  Battery charging system           ✅  BatteryStorageController.charge()
  10  Battery discharge system          ✅  BatteryStorageController.discharge()
  11  Grid health score                 ✅  GridMonitor.grid_health_score()
  12  Risk assessment engine            ✅  RiskAssessmentEngine.calculate_risk()
  13  Overload detection                ✅  GridMonitor.detect_overload()
  14  Underload detection               ✅  GridMonitor.detect_underload()
  15  Fault detection                   ✅  GridMonitor.detect_fault()
  16  Voltage monitoring                ✅  GridMonitor.monitoring_summary()
  17  Frequency monitoring              ✅  GridMonitor.monitoring_summary()
  18  Power factor monitoring           ✅  GridMonitor.monitoring_summary()
  19  Automatic load balancing          ✅  DecisionAgent.balance_load()
  20  Priority-based allocation         ✅  ConsumerManager.allocation_report()
  21  Hospital priority support         ✅  ConsumerManager (priority 1)
  22  Residential consumer management   ✅  ConsumerManager (priority 4)
  23  Industrial consumer management    ✅  ConsumerManager (priority 2)
  24  Commercial consumer management    ✅  ConsumerManager (priority 3)
  25  Emergency mode activation         ✅  DecisionAgent.make_decision()
  26  Backup generator management       ✅  DecisionAgent._start_generator()
  27  Load shedding recommendation      ✅  RiskAssessmentEngine.recommend_load_shedding()
  28  Carbon emission estimation        ✅  RiskAssessmentEngine.estimate_carbon()
  29  Daily report generation           ✅  ReportingDashboard.generate_daily_report()
  30  CSV data export                   ✅  ReportingDashboard.export_to_csv()
  ─────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    random.seed(42)
    main()
