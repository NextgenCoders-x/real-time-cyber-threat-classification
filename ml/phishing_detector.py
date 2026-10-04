import numpy as np

class PhishingDetector:
    """
    Lightweight Phishing Detector.
    Uses basic network/DNS features to detect potential phishing traffic.
    """
    def __init__(self):
        # We look for features like high DNS queries for suspicious domains,
        # unusual payload entropies, or specific ports often used in phishing.
        self.phishing_features = [
            "dns_query_rate",
            "tls_is_self_signed",
            "tls_cert_validity_days"
        ]
        self.confidence_threshold = 0.70

    def predict(self, features: dict) -> dict | None:
        """
        Returns Phishing alert if suspicious features are present.
        """
        dns_rate = float(features.get("dns_query_rate", 0.0))
        self_signed = float(features.get("tls_is_self_signed", 0.0))
        validity = float(features.get("tls_cert_validity_days", 365.0))

        # Heuristic rules for phishing
        # If there's a burst of DNS queries to suspicious self-signed or very short validity certs
        is_phishing = False
        conf = 0.0
        
        if self_signed == 1.0 and validity < 10.0:
            is_phishing = True
            conf = 0.85
        elif dns_rate > 15.0 and validity < 30.0:
            is_phishing = True
            conf = 0.75

        if is_phishing and conf >= self.confidence_threshold:
            return {
                "label": "phishing",
                "severity": "HIGH",
                "confidence": conf,
                "anomaly": True,
                "anomaly_score": -0.5,
                "probabilities": {"phishing": conf, "normal": 1.0 - conf},
                "source": "phishing_detector"
            }
        return None
