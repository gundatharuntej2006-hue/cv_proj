import numpy as np


class TemperatureScaler:
    def __init__(self, temperature: float = 1.25):
        self.temperature = temperature

    def calibrate(self, raw_prob: float) -> float:
        logit = np.log(max(raw_prob, 1e-6) / max(1.0 - raw_prob, 1e-6))
        scaled_logit = logit / self.temperature
        return float(1.0 / (1.0 + np.exp(-scaled_logit)))
