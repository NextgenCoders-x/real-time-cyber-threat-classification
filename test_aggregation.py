import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from ml.trainer import NIDSTrainer
from ml.detector import NIDSDetector
from ml.explainer import NIDSExplainer
from core.flow_tracker import FlowTracker
from core.packet_capture import PacketCapture
from alerts.alert_manager import AlertManager

print("Initializing components for Aggregation Test (Forced Probe)...")

trainer = NIDSTrainer()
trainer.load()
detector = NIDSDetector(trainer)
explainer = NIDSExplainer(trainer)
explainer.build()

alert_mgr = AlertManager()
tracker = FlowTracker()

def on_packet(pkt):
    alert_mgr.record_packet()
    tracker.process_packet(pkt)
    
    for flow, feats in tracker.collect_expired():
        det = detector.predict(feats)
        exp = explainer.explain(feats)
        alert_mgr.add(flow, det, exp)

pc = PacketCapture(callback=on_packet)
pc.start()

print("Force-injecting a 5-second probe attack...")
pc._inject_attack("probe", 5.0)

time.sleep(6)
print("Probe finished. Fast-forwarding 31 seconds for flows to expire...")
time.sleep(31)

for flow, feats in tracker.collect_expired():
    det = detector.predict(feats)
    exp = explainer.explain(feats)
    alert_mgr.add(flow, det, exp)

pc.stop()

print("\n--- FINAL ALERTS ---")
alerts = alert_mgr.recent_alerts(500)
agg_scans = [a for a in alerts if a['label'] == 'VULN SCAN (AGGREGATED)']
ets = [a for a in alerts if a['raw_label'] == 'emerging_threat']
known = [a for a in alerts if a['severity'] in ['HIGH', 'CRITICAL'] and 'AGGREGATED' not in a['label']]

print(f"Total Aggregated Scans: {len(agg_scans)}")
if agg_scans:
    print(f"Sample Aggregated Scan: {agg_scans[0]['src']} -> {agg_scans[0]['dst']} | {agg_scans[0]['explanation']['top_features'][0]['name']}")

print(f"Total Emerging Threats (should be 0 or very small): {len(ets)}")
print(f"Total Known Threats (non-scan): {len(known)}")
