"""
alerts/alert_manager.py — Classifies detections into alerts,
manages the alert queue, and tracks statistics.
"""

import uuid
import time
from collections import deque, defaultdict
from datetime import datetime

# MITRE ATT&CK mapper (imported lazily to avoid circular import)
_mitre_mapper = None
def _get_mapper():
    global _mitre_mapper
    if _mitre_mapper is None:
        try:
            from mitre.mitre_mapper import MITREMapper
            _mitre_mapper = MITREMapper()
        except Exception:
            _mitre_mapper = None
    return _mitre_mapper


SEVERITY_COLOUR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f59e0b",
    "MEDIUM":   "#3b82f6",
    "LOW":      "#10b981",
    "CLEAN":    "#6b7280",
}

# Colour per attack class (for dashboard donut + badges)
CLASS_COLOUR = {
    "normal":       "#10b981",   # green
    "dos":          "#ef4444",   # red
    "ddos":         "#dc2626",   # deep red
    "port_scan":    "#3b82f6",   # blue
    "vuln_scan":    "#60a5fa",   # light blue
    "brute_force":  "#f59e0b",   # amber
    "exploit":      "#8b5cf6",   # purple
    "web_attack":   "#ec4899",   # pink
    "infiltration": "#f97316",   # orange
    "exfiltration": "#14b8a6",   # teal
    "emerging_threat": "#ff00ff", # magenta
    "phishing":     "#0ea5e9",   # light blue
    "ransomware":   "#831843",   # dark pink
    "encrypted_c2": "#4c1d95",   # deep purple
    "dga":          "#701a75",   # purple
}

MAX_ALERTS = 500      # rolling window kept in memory


class Alert:
    def __init__(self, flow, detection: dict, explanation: dict):
        self.id          = str(uuid.uuid4())[:8]
        self.ts          = time.time()
        self.timestamp   = datetime.now().strftime("%H:%M:%S")
        self.src_ip      = flow.src_ip
        self.dst_ip      = flow.dst_ip
        self.src_port    = flow.src_port
        self.dst_port    = flow.dst_port
        self.protocol    = ["TCP", "UDP", "ICMP"][min(flow.protocol, 2)]
        self.label       = detection["label"]
        self.severity    = detection["severity"]
        self.confidence  = detection["confidence"]
        self.anomaly     = detection["anomaly"]
        self.explanation = explanation
        self.colour      = SEVERITY_COLOUR.get(self.severity, "#6b7280")
        self.duration    = round(flow.duration, 2)
        self.bytes       = flow.src_bytes + flow.dst_bytes
        
        # Dashboard label logic for Malware
        if self.label in ["ransomware", "encrypted_c2", "dga"]:
            self.dashboard_label = "MALWARE"
        elif self.label == "emerging_threat":
            self.dashboard_label = "EMERGING THREAT"
        else:
            self.dashboard_label = self.label.upper()

        # MITRE ATT&CK enrichment
        mapper = _get_mapper()
        self.mitre = mapper.map(self.label) if mapper else {}
        self.mitre_badge = mapper.format_badge(self.label) if mapper else {}

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "timestamp":   self.timestamp,
            "src":         f"{self.src_ip}:{self.src_port}",
            "dst":         f"{self.dst_ip}:{self.dst_port}",
            "protocol":    self.protocol,
            "label":       self.dashboard_label,
            "raw_label":   self.label,
            "severity":    self.severity,
            "confidence":  f"{self.confidence * 100:.1f}%",
            "anomaly":     self.anomaly,
            "colour":      self.colour,
            "duration":    self.duration,
            "bytes":       self.bytes,
            "explanation": self.explanation,
            "mitre":       self.mitre,
            "mitre_badge": self.mitre_badge,
        }

class AlertManager:
    def __init__(self):
        self._alerts: deque[Alert]      = deque(maxlen=MAX_ALERTS)
        self._stats                     = defaultdict(int)
        self._attack_counts             = defaultdict(int)
        self._src_ip_counts             = defaultdict(int)
        self._packets_per_sec           = deque(maxlen=60)  # 60-sec window
        self._pkt_ts_window: list[float] = []
        self._scan_tracker              = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def _check_scan(self, flow):
        now = time.time()
        key = (flow.src_ip, flow.dst_ip)
        
        if key not in self._scan_tracker:
            self._scan_tracker[key] = {
                "start_ts": now,
                "last_ts": now,
                "ports": {flow.dst_port},
                "flows": 1,
                "alert": None,
            }
            return False, None
            
        state = self._scan_tracker[key]
        
        SCAN_WINDOW = 30.0
        SCAN_PORT_THRESHOLD = 20
        
        if now - state["last_ts"] > SCAN_WINDOW:
            state["start_ts"] = now
            state["last_ts"] = now
            state["ports"] = {flow.dst_port}
            state["flows"] = 1
            state["alert"] = None
            return False, None
            
        state["last_ts"] = now
        state["ports"].add(flow.dst_port)
        state["flows"] += 1
        
        if len(state["ports"]) >= SCAN_PORT_THRESHOLD:
            return True, state
            
        return False, None

    def _purge_scan_micro_flows(self, src_ip, dst_ip, start_ts):
        new_alerts = deque(maxlen=MAX_ALERTS)
        for a in self._alerts:
            if a.src_ip == src_ip and a.dst_ip == dst_ip and a.ts >= start_ts - 1.0:
                self._stats[a.severity] = max(0, self._stats[a.severity] - 1)
                self._attack_counts[a.label] = max(0, self._attack_counts[a.label] - 1)
                self._src_ip_counts[a.src_ip] = max(0, self._src_ip_counts[a.src_ip] - 1)
            else:
                new_alerts.append(a)
        self._alerts = new_alerts

    def add(self, flow, detection: dict, explanation: dict) -> Alert | None:
        """Create and store an alert. Returns None if traffic is clean."""
        self._stats["total_flows"] += 1

        is_scan, scan_state = self._check_scan(flow)
        
        if is_scan:
            if scan_state["alert"] is None:
                self._purge_scan_micro_flows(flow.src_ip, flow.dst_ip, scan_state["start_ts"])
                
                scan_detection = {
                    "label": "vuln_scan",
                    "severity": "HIGH",
                    "confidence": 0.99,
                    "anomaly": "Port Scan Pattern Detected"
                }
                scan_explanation = {
                    "top_features": [{"name": f"{len(scan_state['ports'])} ports scanned"}]
                }
                scan_alert = Alert(flow, scan_detection, scan_explanation)
                scan_alert.dashboard_label = "VULN SCAN (AGGREGATED)"
                scan_state["alert"] = scan_alert
                self._alerts.appendleft(scan_alert)
                
                self._stats[scan_alert.severity] += 1
                self._attack_counts[scan_alert.label] += 1
                self._src_ip_counts[scan_alert.src_ip] += 1
                return scan_alert
            else:
                alert = scan_state["alert"]
                alert.explanation["top_features"] = [{"name": f"{len(scan_state['ports'])} ports scanned ({scan_state['flows']} attempts)"}]
                alert.duration = round(scan_state["last_ts"] - scan_state["start_ts"], 2)
                alert.timestamp = datetime.now().strftime("%H:%M:%S")
                return None

        if detection["severity"] == "CLEAN":
            self._stats["clean"] += 1
            return None

        alert = Alert(flow, detection, explanation)
        self._alerts.appendleft(alert)
        self._stats[alert.severity]   += 1
        self._attack_counts[alert.label] += 1
        self._src_ip_counts[alert.src_ip] += 1
        return alert

    def record_packet(self):
        """Call for every processed packet to track packet/sec rate."""
        now = time.time()
        self._pkt_ts_window = [t for t in self._pkt_ts_window if now - t < 1.0]
        self._pkt_ts_window.append(now)

    def recent_alerts(self, n: int = 50) -> list[dict]:
        return [a.to_dict() for a in list(self._alerts)[:n]]

    def dashboard_stats(self) -> dict:
        total = max(self._stats["total_flows"], 1)
        # Count both explicit "CLEAN" and "LOW" severity (normal) flows as clean traffic
        clean_flows = self._stats["clean"] + self._stats.get("LOW", 0)
        
        return {
            "total_flows":     self._stats["total_flows"],
            "clean":           clean_flows,
            "critical":        self._stats.get("CRITICAL", 0),
            "high":            self._stats.get("HIGH", 0),
            "medium":          self._stats.get("MEDIUM", 0),
            "low":             self._stats.get("LOW", 0),
            "threat_pct":      round((total - clean_flows) / total * 100, 1),
            "clean_pct":       round(clean_flows / total * 100, 1),
            "low":             self._stats.get("LOW", 0),
            "threat_pct":      round((total - clean_flows) / total * 100, 1),
            "clean_pct":       round(clean_flows / total * 100, 1),
            "packets_per_sec": len(self._pkt_ts_window),
            "attack_types":    self._get_grouped_attacks(),
            "top_sources":     self._top_sources(5),
            "total_alerts":    len(self._alerts),
            "emerging_threats": self._attack_counts.get("emerging_threat", 0),
            "known_threats":   sum(v for k, v in self._attack_counts.items() if k != "emerging_threat"),
        }

    def _get_grouped_attacks(self) -> dict:
        grouped = defaultdict(int)
        for k, v in self._attack_counts.items():
            if k in ["ransomware", "encrypted_c2", "dga"]:
                grouped["malware"] += v
            else:
                grouped[k] += v
        return dict(grouped)

    def clear(self):
        self._alerts.clear()
        self._stats.clear()
        self._attack_counts.clear()
        self._src_ip_counts.clear()

    def all_alerts_for_report(self) -> list[dict]:
        return [a.to_dict() for a in self._alerts]

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _top_sources(self, n: int) -> list[dict]:
        sorted_src = sorted(
            self._src_ip_counts.items(),
            key=lambda x: x[1], reverse=True
        )[:n]
        return [{"ip": ip, "count": cnt} for ip, cnt in sorted_src]
