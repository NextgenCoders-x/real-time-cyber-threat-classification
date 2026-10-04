"""
core/flow_tracker.py — Aggregates raw packet events into network flows
and extracts ML-ready features from each completed flow.

FEATURE CONTRACT (authoritative 24-feature schema):
=============================================================================
Feature                  | Meaning                              | Formula
=============================================================================
duration                 | Flow lifetime (sec)                  | last_seen - start_time
protocol_type            | Network protocol                     | 0=TCP,1=UDP,2=ICMP
src_bytes                | Bytes sent by source                 | cumulative
dst_bytes                | Bytes sent by destination            | cumulative
wrong_fragment           | IP fragment errors                   | cumulative count
urgent                   | URG flag count                       | cumulative count
count                    | Connections to same dst host, 2s wnd | KDD-compatible, connection-level
srv_count                | Connections to same dst service, 2s  | KDD-compatible, connection-level
serror_rate              | SYN-error fraction of this flow      | syn_pkts / pkt_count
rerror_rate              | RST-error fraction of this flow      | rst_pkts / pkt_count
same_srv_rate            | Fraction of `count` to same service  | srv_count / count
diff_srv_rate            | 1 - same_srv_rate                    | 1 - same_srv_rate
dst_host_count           | Distinct destination hosts seen      | len(dst_host_log)
dst_host_srv_count       | Distinct services on this dst host   | len(ports_to_dst)
dst_host_same_srv_rate   | Fraction of dst ports = this port    | same_port / total_ports
dst_host_diff_srv_rate   | 1 - dst_host_same_srv_rate           | complement
dst_host_serror_rate     | SYN-error rate across dst hosts      | approximation (0.0)
packet_rate              | Packets per second (this flow)       | pkt_count / duration
byte_rate                | Bytes per second  (this flow)        | total_bytes / duration
flag_syn_ratio           | SYN packets / total packets          | syn_count / pkt_count
flag_fin_ratio           | FIN packets / total packets          | fin_count / pkt_count
flag_rst_ratio           | RST packets / total packets          | rst_count / pkt_count
port_number              | Destination port number              | numeric
is_well_known_port       | dst_port < 1024                      | bool -> 0.0 or 1.0
=============================================================================

IMPORTANT — count/srv_count semantics (KDD-compatible):
  These are CONNECTION-level counts within a 2-second window to the same
  destination host/service, NOT packet-level counts.  The FlowTracker
  records one entry per new flow (connection start), so count reflects
  how many concurrent or recent connections go to the same host,
  exactly matching the KDD 'count' and 'srv_count' semantics.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

FLOW_TIMEOUT  = 30.0   # seconds of inactivity before a flow is finalised
CONN_WINDOW   = 2.0    # KDD-compatible 2-second connection counting window


@dataclass
class Flow:
    src_ip:    str
    dst_ip:    str
    src_port:  int
    dst_port:  int
    protocol:  int      # 0=TCP  1=UDP  2=ICMP

    # Counters updated as packets arrive
    start_time:    float = field(default_factory=time.time)
    last_seen:     float = field(default_factory=time.time)
    pkt_count:     int   = 0
    src_bytes:     int   = 0
    dst_bytes:     int   = 0
    syn_count:     int   = 0
    fin_count:     int   = 0
    rst_count:     int   = 0
    urg_count:     int   = 0
    wrong_frags:   int   = 0

    def update(self, pkt_size: int, is_src: bool, flags: dict):
        self.pkt_count  += 1
        self.last_seen   = time.time()
        if is_src:
            self.src_bytes += pkt_size
        else:
            self.dst_bytes += pkt_size
        self.syn_count  += int(flags.get("SYN", False))
        self.fin_count  += int(flags.get("FIN", False))
        self.rst_count  += int(flags.get("RST", False))
        self.urg_count  += int(flags.get("URG", False))

    @property
    def duration(self) -> float:
        """Seconds between first and last packet, minimum 0.001 to avoid /0."""
        return max(self.last_seen - self.start_time, 0.001)

    def is_expired(self) -> bool:
        return (time.time() - self.last_seen) > FLOW_TIMEOUT

    def to_features(self, global_stats: dict) -> dict:
        """Convert flow into an ML feature vector.

        count, srv_count, same_srv_rate, diff_srv_rate are taken from
        global_stats (KDD-compatible, connection-level 2s window) — they
        must NOT be computed here from raw packet counts.
        """
        duration   = self.duration
        pkt_cnt    = max(self.pkt_count, 1)
        # Use a minimum 0.5s floor for RATE calculations only.
        # This prevents 1-2 packet flows from producing astronomical
        # packet_rate/byte_rate values (e.g., 1pkt/0.001s = 1000 pkt/s).
        # The "duration" feature itself is NOT clamped — it stays accurate.
        rate_dur   = max(duration, 0.5)
        pkt_rate   = pkt_cnt / rate_dur
        byte_rate  = (self.src_bytes + self.dst_bytes) / rate_dur

        return {
            # ── Connection-level features ─────────────────────────────────
            "duration":               duration,
            "protocol_type":          float(self.protocol),
            "src_bytes":              float(self.src_bytes),
            "dst_bytes":              float(self.dst_bytes),
            "wrong_fragment":         float(self.wrong_frags),
            "urgent":                 float(self.urg_count),

            # ── KDD-compatible connection-window stats (from global_stats) ─
            "count":                  float(global_stats.get("count",         1)),
            "srv_count":              float(global_stats.get("srv_count",     1)),
            "serror_rate":            self.syn_count / pkt_cnt,
            "rerror_rate":            self.rst_count / pkt_cnt,
            "same_srv_rate":          float(global_stats.get("same_srv_rate", 1.0)),
            "diff_srv_rate":          float(global_stats.get("diff_srv_rate", 0.0)),

            # ── Destination-host stats ────────────────────────────────────
            "dst_host_count":         float(global_stats.get("dst_host_count",         1)),
            "dst_host_srv_count":     float(global_stats.get("dst_host_srv_count",     1)),
            "dst_host_same_srv_rate": float(global_stats.get("dst_host_same_srv_rate", 1.0)),
            "dst_host_diff_srv_rate": float(global_stats.get("dst_host_diff_srv_rate", 0.0)),
            "dst_host_serror_rate":   float(global_stats.get("dst_host_serror_rate",   0.0)),

            # ── Per-flow rate features ────────────────────────────────────
            "packet_rate":            pkt_rate,
            "byte_rate":              byte_rate,

            # ── TCP flag ratios ───────────────────────────────────────────
            "flag_syn_ratio":         self.syn_count / pkt_cnt,
            "flag_fin_ratio":         self.fin_count / pkt_cnt,
            "flag_rst_ratio":         self.rst_count / pkt_cnt,

            # ── Port features ─────────────────────────────────────────────
            "port_number":            float(self.dst_port),
            "is_well_known_port":     float(self.dst_port < 1024),
        }


class FlowTracker:
    """Tracks all active network flows and emits completed flows with features."""

    def __init__(self):
        self._flows:         dict[tuple, Flow]        = {}
        self._history:       list[dict]               = []
        self._dst_host_log:  dict[str, set]           = defaultdict(set)

        # KDD-compatible per-destination connection log isolated by source:
        # {(src_ip, dst_ip): [(timestamp_of_connection_start, dst_port), ...]}
        # One entry per NEW flow creation (not per packet).
        self._src_dst_conn_log: dict[tuple[str, str], list] = defaultdict(list)

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_packet(self, pkt_info: dict) -> "Flow | None":
        """Feed one packet. Returns a completed Flow if one just expired, else None."""
        key     = self._flow_key(pkt_info)
        rev_key = self._flow_key(pkt_info, reverse=True)

        now = time.time()
        dst = pkt_info.get("dst_ip", "")
        src = pkt_info.get("src_ip", "0.0.0.0")

        # Determine if this packet starts a new flow
        is_new_flow = (key not in self._flows) and (rev_key not in self._flows)

        if key in self._flows:
            flow   = self._flows[key]
            is_src = True
        elif rev_key in self._flows:
            flow   = self._flows[rev_key]
            key    = rev_key
            is_src = False
        else:
            # New flow — create it
            flow = Flow(
                src_ip   = src,
                dst_ip   = dst,
                src_port = pkt_info.get("src_port", 0),
                dst_port = pkt_info.get("dst_port", 0),
                protocol = pkt_info.get("protocol", 0),
            )
            self._flows[key] = flow
            is_src = True

        flow.update(
            pkt_size = pkt_info.get("size", 0),
            is_src   = is_src,
            flags    = pkt_info.get("flags", {}),
        )

        # Track destination-port history (legacy behavior for dst_host_count)
        self._dst_host_log[dst].add(pkt_info.get("dst_port", 0))

        # Record new CONNECTION start for KDD-compatible count/srv_count
        if is_new_flow:
            self._src_dst_conn_log[(src, dst)].append((now, pkt_info.get("dst_port", 0)))

        return self._expire_flow(key)

    def collect_expired(self) -> list[tuple["Flow", dict]]:
        """Return and remove all expired flows as (flow, features) pairs."""
        expired_keys = [k for k, f in list(self._flows.items()) if f.is_expired()]
        results = []
        for key in expired_keys:
            flow  = self._flows.pop(key)
            stats = self._global_stats(flow)
            results.append((flow, flow.to_features(stats)))
        return results

    def active_count(self) -> int:
        return len(self._flows)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _expire_flow(self, key: tuple) -> "Flow | None":
        flow = self._flows.get(key)
        if flow and flow.is_expired() and flow.pkt_count > 0:
            return self._flows.pop(key)
        return None

    def _global_stats(self, flow: Flow) -> dict:
        """Compute KDD-compatible global connection statistics for a flow.

        To prevent 'collateral flow tainting' from simultaneous attacks,
        the context window is isolated to the specific (src_ip, dst_ip) pair.
        """
        now = time.time()
        dst = flow.dst_ip
        src = flow.src_ip
        c_key = (src, dst)

        # Prune stale entries outside the 2-second window
        self._src_dst_conn_log[c_key] = [
            (ts, port) for ts, port in self._src_dst_conn_log.get(c_key, [])
            if now - ts < CONN_WINDOW
        ]
        dst_conns = self._src_dst_conn_log[c_key]

        # Context-isolated connection counts
        count      = max(len(dst_conns), 1)
        srv_conns  = [x for x in dst_conns if x[1] == flow.dst_port]
        srv_count  = max(len(srv_conns), 1)

        same_srv_rate = srv_count / count
        diff_srv_rate = 1.0 - same_srv_rate

        return {
            "count":                  count,
            "srv_count":              srv_count,
            "same_srv_rate":          same_srv_rate,
            "diff_srv_rate":          diff_srv_rate,
            "dst_host_count":         len(self._dst_host_log),

            # In KDD, dst_host_srv_count is the count of connections to the SAME service.
            # So we approximate it using our 2s srv_count.
            "dst_host_srv_count":     srv_count,
            # Approximate the 100-conn window with our current connection rates
            "dst_host_same_srv_rate": same_srv_rate,
            "dst_host_diff_srv_rate": diff_srv_rate,
            "dst_host_serror_rate":   0.0,   # approximated as 0
        }

    @staticmethod
    def _flow_key(pkt: dict, reverse: bool = False) -> tuple:
        if reverse:
            return (pkt.get("dst_ip"), pkt.get("src_ip"),
                    pkt.get("dst_port"), pkt.get("src_port"),
                    pkt.get("protocol"))
        return (pkt.get("src_ip"), pkt.get("dst_ip"),
                pkt.get("src_port"), pkt.get("dst_port"),
                pkt.get("protocol"))
