"""Post-hoc probability calibration via Temperature Scaling (Platt scaling on temperature)."""

from typing import Union
import numpy as np
from scipy.optimize import minimize_scalar, minimize


class TemperatureScaler:
    """
    Fits and applies temperature scaling to logits/probabilities to minimize NLL on the development split.
    Supports both binary logits and 3-class logits.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(np.clip(z_shifted, -30.0, 30.0))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def _prob_to_logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
        p_clipped = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        return np.log(p_clipped / (1.0 - p_clipped))

    def fit_binary(self, logits_or_probs: np.ndarray | list, labels: np.ndarray | list, is_prob: bool = False) -> float:
        """
        Fits optimal temperature T for binary classification by minimizing binary cross-entropy.
        
        Args:
            logits_or_probs: raw logits or uncalibrated probabilities
            labels: binary labels (0 or 1)
            is_prob: if True, converts probabilities to log-odds logits first
            
        Returns:
            fitted temperature T
        """
        y = np.asarray(labels, dtype=float)
        if is_prob:
            z = self._prob_to_logit(logits_or_probs)
        else:
            z = np.asarray(logits_or_probs, dtype=float)

        def nll(t: float) -> float:
            scaled_z = z / max(t, 1e-4)
            p = self._sigmoid(scaled_z)
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
            return float(loss)

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    def fit_multiclass(self, logits: np.ndarray, labels: np.ndarray | list) -> float:
        """
        Fits optimal temperature T for multi-class classification by minimizing categorical cross-entropy.
        """
        y = np.asarray(labels, dtype=int)
        z = np.asarray(logits, dtype=float)
        n_samples = len(y)

        def nll(t_arr: np.ndarray) -> float:
            t = float(t_arr[0])
            scaled_z = z / max(t, 1e-4)
            probs = self._softmax(scaled_z)
            eps = 1e-12
            correct_log_probs = np.log(probs[np.arange(n_samples), y] + eps)
            return float(-np.mean(correct_log_probs))

        res = minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def calibrate(self, logits_or_probs: Union[float, np.ndarray, list], is_prob: bool = True) -> np.ndarray | float:
        """
        Applies temperature scaling to convert raw logits/probabilities to calibrated probabilities.
        """
        is_scalar = isinstance(logits_or_probs, (int, float))
        arr = np.asarray(logits_or_probs, dtype=float)

        if arr.ndim == 0:
            arr = np.array([arr])

        if arr.ndim == 1 and not (is_prob and arr.shape[0] == 3 and not is_scalar):
            # Binary probabilities or logits
            if is_prob:
                z = self._prob_to_logit(arr)
            else:
                z = arr
            calib = self._sigmoid(z / self.temperature)
        elif arr.ndim >= 1 and arr.shape[-1] == 3:
            # 3-Class logits or probabilities
            if is_prob:
                z = np.log(np.clip(arr, 1e-6, 1.0))
            else:
                z = arr
            calib = self._softmax(z / self.temperature)
        else:
            if is_prob:
                z = self._prob_to_logit(arr)
            else:
                z = arr
            calib = self._sigmoid(z / self.temperature)

        if is_scalar:
            return float(calib[0])
        return calib
