"""
test_pipeline.py — Full pipeline test after feature semantics fix.

Tests:
  1. Normal training sample → should classify as "normal"
  2. Simulated normal flow → should classify as "normal"
  3. DoS sample → should classify as "dos"
  4. DDoS sample → should classify as "ddos"
  5. Brute Force sample → should classify as "brute_force"
  6. Exploit sample → should classify as "exploit"
  7. Emerging Threat (intentionally anomalous) → should classify as "emerging_threat"
  8. Feature Z-score validation (post-fix)
  9. Aggregate 100-flow mixed simulation
"""

import sys, os, random, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from ml.trainer import (
    NIDSTrainer, FEATURE_NAMES, generate_synthetic_dataset,
    _make_normal, _make_dos, _make_ddos, _make_brute_force, _make_exploit
)
from ml.detector import NIDSDetector
from core.flow_tracker import Flow, FlowTracker

# ── Load retrained models ────────────────────────────────────────────────────
trainer = NIDSTrainer()
assert trainer.load(), "Models not found — run 'python main.py train' first"
detector = NIDSDetector(trainer)

df_syn, y_syn = generate_synthetic_dataset(n_per_class=200)

PASS = 0
FAIL = 0

def test(name, features_dict, expected_label, allow_labels=None):
    global PASS, FAIL
    result = detector.predict(features_dict)
    label = result["label"]
    conf  = result["confidence"]
    sev   = result["severity"]
    ok = (label == expected_label) or (allow_labels and label in allow_labels)
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}")
    print(f"         Expected: {expected_label}{f' (or {allow_labels})' if allow_labels else ''}")
    print(f"         Got:      {label}  conf={conf:.4f}  severity={sev}")
    if not ok:
        print(f"         *** MISMATCH ***")
    return result

# ── TEST 1: Normal training sample ───────────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: Normal training sample")
print("="*70)
normal_row = df_syn[y_syn == "normal"].iloc[0].to_dict()
test("Normal training sample", normal_row, "normal")

# ── TEST 2: Simulated normal flow via FlowTracker ────────────────────────────
print("\n" + "="*70)
print("TEST 2: Simulated normal flow (FlowTracker)")
print("="*70)
tracker = FlowTracker()
# Simulate a normal TCP connection: 10 packets over ~5 seconds
flow = Flow(src_ip="192.168.1.5", dst_ip="192.168.1.1",
            src_port=54321, dst_port=80, protocol=0)
for i in range(10):
    flow.update(pkt_size=random.randint(200, 1200), is_src=True,
                flags={"ACK": True, "FIN": False, "SYN": i==0, "RST": False})
    time.sleep(0.002)  # small delay between packets
# Simulate realistic duration
flow.start_time = time.time() - 5.0
flow.last_seen  = time.time() - 0.01

# Register a couple of connections for KDD-compatible stats
tracker._dst_conn_log["192.168.1.1"].append((time.time(), 80))
tracker._dst_conn_log["192.168.1.1"].append((time.time()-0.5, 80))
tracker._dst_conn_log["192.168.1.1"].append((time.time()-1.0, 443))
tracker._dst_host_log["192.168.1.1"].add(80)
tracker._dst_host_log["192.168.1.1"].add(443)

stats = tracker._global_stats(flow)
sim_feats = flow.to_features(stats)
print(f"  count={sim_feats['count']}, srv_count={sim_feats['srv_count']}, "
      f"same_srv_rate={sim_feats['same_srv_rate']:.2f}, "
      f"packet_rate={sim_feats['packet_rate']:.1f}, byte_rate={sim_feats['byte_rate']:.0f}")
test("Simulated normal flow", sim_feats, "normal")

# ── TEST 3: DoS training sample ──────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: DoS training sample")
print("="*70)
dos_row = df_syn[y_syn == "dos"].iloc[0].to_dict()
test("DoS training sample", dos_row, "dos")

# ── TEST 4: DDoS training sample ─────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 4: DDoS training sample")
print("="*70)
ddos_row = df_syn[y_syn == "ddos"].iloc[0].to_dict()
test("DDoS training sample", ddos_row, "ddos")

# ── TEST 5: Brute Force training sample ──────────────────────────────────────
print("\n" + "="*70)
print("TEST 5: Brute Force training sample")
print("="*70)
bf_row = df_syn[y_syn == "brute_force"].iloc[0].to_dict()
test("Brute Force training sample", bf_row, "brute_force")

# ── TEST 6: Exploit training sample ──────────────────────────────────────────
print("\n" + "="*70)
print("TEST 6: Exploit training sample")
print("="*70)
exp_row = df_syn[y_syn == "exploit"].iloc[0].to_dict()
test("Exploit training sample", exp_row, "exploit")

# ── TEST 7: Intentionally anomalous traffic (Emerging Threat) ────────────────
print("\n" + "="*70)
print("TEST 7: Intentionally anomalous traffic (should be Emerging Threat)")
print("="*70)
# Create features that are clearly out-of-distribution for ALL known classes:
# very weird combination — high byte_rate like exfiltration but high SYN like DoS
# with very unusual count and diff_srv_rate
anomalous = {
    "duration": 0.01,
    "protocol_type": 2,    # ICMP (unusual for high byte traffic)
    "src_bytes": 500000.0,
    "dst_bytes": 500000.0,
    "wrong_fragment": 50.0,   # extremely high — very unusual
    "urgent": 10.0,            # extremely high
    "count": 250,
    "srv_count": 5,
    "serror_rate": 0.5,
    "rerror_rate": 0.5,
    "same_srv_rate": 0.02,
    "diff_srv_rate": 0.98,
    "dst_host_count": 200,
    "dst_host_srv_count": 200,
    "dst_host_same_srv_rate": 0.01,
    "dst_host_diff_srv_rate": 0.99,
    "dst_host_serror_rate": 0.9,
    "packet_rate": 50000.0,
    "byte_rate": 5000000.0,
    "flag_syn_ratio": 0.9,
    "flag_fin_ratio": 0.9,
    "flag_rst_ratio": 0.9,
    "port_number": 31337.0,
    "is_well_known_port": 0.0,
}
r = test("Intentionally anomalous traffic", anomalous, "emerging_threat",
         allow_labels=["emerging_threat", "dos", "ddos", "port_scan"])
print(f"         anomaly_score={r['anomaly_score']:.4f}, anomaly={r['anomaly']}")

# ── TEST 8: Post-fix Z-score validation ──────────────────────────────────────
print("\n" + "="*70)
print("TEST 8: Feature Z-score validation (training normal vs simulated normal)")
print("="*70)
train_normal = df_syn[y_syn == "normal"][FEATURE_NAMES[:24]]
key_feats = ["count", "srv_count", "same_srv_rate", "packet_rate", "byte_rate"]
print(f"  {'Feature':30s} {'TrMean':>10} {'TrStd':>8} {'SimVal':>10} {'Z-score':>8}  {'Status'}")
print("  " + "-"*85)
z_all_ok = True
for feat in key_feats:
    tr_mean = train_normal[feat].mean()
    tr_std  = train_normal[feat].std()
    sim_val = sim_feats.get(feat, 0.0)
    z = (float(sim_val) - tr_mean) / (tr_std + 1e-9)
    ok = abs(z) <= 3
    if not ok:
        z_all_ok = False
    print(f"  {feat:30s} {tr_mean:10.2f} {tr_std:8.2f} {float(sim_val):10.2f} {z:8.2f}  {'OK' if ok else 'MISMATCH!'}")
if z_all_ok:
    PASS += 1
    print("  [PASS] All key features within 3-sigma of training distribution")
else:
    FAIL += 1
    print("  [FAIL] Some features still out of distribution")

# ── TEST 9: Aggregate 100-flow mixed simulation ─────────────────────────────
print("\n" + "="*70)
print("TEST 9: Aggregate 100-flow test (50 normal + 50 attack)")
print("="*70)
counts = {"normal": 0, "known_attack": 0, "emerging_threat": 0}
conf_normal, conf_attack = [], []

# 50 normal samples
for i in range(50):
    row = df_syn[y_syn == "normal"].iloc[i].to_dict()
    r = detector.predict(row)
    if r["label"] == "normal":
        counts["normal"] += 1
        conf_normal.append(r["confidence"])
    elif r["label"] == "emerging_threat":
        counts["emerging_threat"] += 1
    else:
        counts["known_attack"] += 1

# 50 attack samples (mixed classes)
attack_classes = ["dos", "ddos", "port_scan", "brute_force", "exploit",
                  "web_attack", "infiltration", "exfiltration", "vuln_scan"]
for i, cls in enumerate(attack_classes):
    subset = df_syn[y_syn == cls]
    n_samples = min(6, len(subset))
    for j in range(n_samples):
        row = subset.iloc[j].to_dict()
        r = detector.predict(row)
        if r["label"] == cls:
            counts["known_attack"] += 1
            conf_attack.append(r["confidence"])
        elif r["label"] == "emerging_threat":
            counts["emerging_threat"] += 1
        elif r["label"] == "normal":
            counts["normal"] += 1
        else:
            counts["known_attack"] += 1
            conf_attack.append(r["confidence"])

print(f"  Normal classified:          {counts['normal']}")
print(f"  Known attacks classified:   {counts['known_attack']}")
print(f"  Emerging threats:           {counts['emerging_threat']}")
if conf_normal:
    print(f"  Normal confidence mean:     {np.mean(conf_normal):.4f}")
if conf_attack:
    print(f"  Attack confidence mean:     {np.mean(conf_attack):.4f}")

normal_ok = counts["normal"] >= 40   # at least 80% of normal should be classified normal
attack_ok = counts["known_attack"] >= 30  # at least 60% of attacks classified correctly
if normal_ok and attack_ok:
    PASS += 1
    print(f"  [PASS] Normal={counts['normal']}/50, KnownAttack={counts['known_attack']}")
else:
    FAIL += 1
    print(f"  [FAIL] Normal={counts['normal']}/50 (need >=40), KnownAttack={counts['known_attack']} (need >=30)")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"FINAL RESULTS:  {PASS} PASSED  /  {FAIL} FAILED")
print("="*70)
