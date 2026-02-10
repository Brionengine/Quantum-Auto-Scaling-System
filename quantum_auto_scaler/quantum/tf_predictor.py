"""
Brion Quantum - TensorFlow Demand Predictor v1.0
===================================================
LSTM-based demand prediction using TensorFlow for time-series forecasting
of system workload. Supports both CPU and TPU execution.

Developed by Brion Quantum AI Team
"""

__version__ = "1.0.1"

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# TensorFlow availability
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    pass


class TensorFlowDemandPredictor:
    """
    LSTM-based demand predictor using TensorFlow.

    Trains on historical workload data to predict future demand,
    enabling proactive scaling decisions.
    """

    def __init__(self, lookback: int = 20, forecast_horizon: int = 5,
                 hidden_units: int = 32, use_tpu: bool = False):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.hidden_units = hidden_units
        self.use_tpu = use_tpu

        self.model = None
        self.is_trained = False
        self.training_history: List[Dict[str, float]] = []

        # TPU setup
        self._tpu_resolver = None
        self._strategy = None

        if TF_AVAILABLE:
            self._setup_device()
            self._build_model()

    def _setup_device(self):
        """Setup TensorFlow compute device (CPU/GPU/TPU)."""
        if not TF_AVAILABLE:
            return

        if self.use_tpu:
            try:
                self._tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(self._tpu_resolver)
                tf.tpu.experimental.initialize_tpu_system(self._tpu_resolver)
                self._strategy = tf.distribute.TPUStrategy(self._tpu_resolver)
            except Exception:
                self._strategy = tf.distribute.get_strategy()
        else:
            self._strategy = tf.distribute.get_strategy()

    def _build_model(self):
        """Build LSTM prediction model."""
        if not TF_AVAILABLE:
            return

        def _create():
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.hidden_units,
                                      input_shape=(self.lookback, 1),
                                      return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(self.hidden_units // 2),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.forecast_horizon)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model

        if self._strategy:
            with self._strategy.scope():
                self.model = _create()
        else:
            self.model = _create()

    def train(self, workload_history: List[float],
              epochs: int = 50, batch_size: int = 16) -> Dict[str, Any]:
        """
        Train the prediction model on historical workload data.

        Args:
            workload_history: Historical utilization values (0-1)
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training summary.
        """
        if not TF_AVAILABLE or self.model is None:
            return {"error": "TensorFlow not available"}

        if len(workload_history) < self.lookback + self.forecast_horizon + 10:
            return {"error": "Insufficient training data"}

        # Prepare sequences
        X, y = self._prepare_sequences(workload_history)

        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )

        self.is_trained = True

        result = {
            "epochs": epochs,
            "final_loss": float(history.history['loss'][-1]),
            "final_mae": float(history.history['mae'][-1]),
            "val_loss": float(history.history.get('val_loss', [0])[-1]),
            "samples": len(X)
        }

        self.training_history.append(result)
        return result

    def predict(self, recent_data: List[float]) -> List[float]:
        """
        Predict future demand from recent data.

        Args:
            recent_data: Last N data points (N >= lookback)

        Returns:
            List of predicted values for next forecast_horizon steps.
        """
        if not TF_AVAILABLE or self.model is None or not self.is_trained:
            return self._simple_predict(recent_data)

        if len(recent_data) < self.lookback:
            return self._simple_predict(recent_data)

        # Prepare input
        X = np.array(recent_data[-self.lookback:]).reshape(1, self.lookback, 1)
        prediction = self.model.predict(X, verbose=0)[0]

        return [float(np.clip(p, 0.0, 1.0)) for p in prediction]

    def _simple_predict(self, data: List[float]) -> List[float]:
        """Simple fallback predictor (exponential smoothing)."""
        if not data:
            return [0.5] * self.forecast_horizon

        alpha = 0.3
        smoothed = data[-1]
        trend = np.mean(np.diff(data[-5:])) if len(data) >= 5 else 0.0

        predictions = []
        for i in range(self.forecast_horizon):
            pred = smoothed + trend * (i + 1)
            predictions.append(float(np.clip(pred, 0.0, 1.0)))

        return predictions

    def _prepare_sequences(self, data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences from raw data."""
        arr = np.array(data)
        X, y = [], []

        for i in range(len(arr) - self.lookback - self.forecast_horizon):
            X.append(arr[i:i + self.lookback])
            y.append(arr[i + self.lookback:i + self.lookback + self.forecast_horizon])

        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)
        return X, y

    def report(self) -> Dict[str, Any]:
        """Report predictor status."""
        return {
            "version": __version__,
            "tensorflow_available": TF_AVAILABLE,
            "is_trained": self.is_trained,
            "use_tpu": self.use_tpu,
            "tpu_connected": self._tpu_resolver is not None,
            "lookback": self.lookback,
            "forecast_horizon": self.forecast_horizon,
            "training_runs": len(self.training_history),
            "latest_training": self.training_history[-1] if self.training_history else None
        }
