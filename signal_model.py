"""Signal Processing models for sports predictions.

Fourier analysis: Detect periodic patterns in team performance.
Wavelet transforms: Multi-scale decomposition revealing trends at different time scales.

Packages: numpy (FFT built-in), PyWavelets (pip install PyWavelets)
"""

import logging
import math
from collections import defaultdict

import numpy as np


class TeamSignalAnalyzer:
    """Fourier + wavelet analysis of a team's performance time series."""

    def __init__(self, min_games=15):
        self.min_games = min_games
        self.margins = []
        self._fourier_features = {}
        self._wavelet_features = {}

    def add_game(self, margin):
        self.margins.append(margin)

    def analyze_fourier(self):
        """Detect dominant frequencies in team performance.

        Returns features about periodic patterns:
        - dominant_period: most common cycle length (in games)
        - spectral_entropy: how spread the energy is (high = noisy, low = periodic)
        - low_freq_power: fraction of energy in long cycles (>5 games)
        """
        if len(self.margins) < self.min_games:
            return {}

        x = np.array(self.margins, dtype=float)
        x = x - np.mean(x)  # Remove DC component

        n = len(x)
        fft_vals = np.fft.rfft(x)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # Skip DC (index 0)
        power = power[1:]
        freqs = freqs[1:]

        if len(power) == 0 or np.sum(power) == 0:
            return {}

        # Normalize power spectrum
        power_norm = power / np.sum(power)

        # Dominant frequency
        dom_idx = np.argmax(power)
        dom_freq = freqs[dom_idx]
        dom_period = 1.0 / dom_freq if dom_freq > 0 else n

        # Spectral entropy (0=pure tone, high=noise)
        power_norm_nz = power_norm[power_norm > 0]
        spectral_entropy = -np.sum(power_norm_nz * np.log2(power_norm_nz))
        max_entropy = np.log2(len(power_norm_nz)) if len(power_norm_nz) > 0 else 1
        norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 1.0

        # Low frequency power (cycles > 5 games)
        low_freq_mask = freqs < 0.2  # period > 5 games
        low_freq_power = np.sum(power[low_freq_mask]) / np.sum(power) if np.sum(power) > 0 else 0.5

        self._fourier_features = {
            "dominant_period": float(dom_period),
            "spectral_entropy": float(norm_entropy),
            "low_freq_power": float(low_freq_power),
            "dominant_amplitude": float(np.sqrt(power[dom_idx])),
        }
        return self._fourier_features

    def analyze_wavelet(self):
        """Multi-scale wavelet decomposition.

        Decomposes team performance into components at different time scales:
        - Detail coefficients at each level reveal patterns at that scale
        - Approximation shows the long-term trend
        """
        if len(self.margins) < self.min_games:
            return {}

        try:
            import pywt
        except ImportError:
            return self._fallback_multiscale()

        x = np.array(self.margins, dtype=float)

        # Determine max decomposition level
        max_level = min(3, pywt.dwt_max_level(len(x), "db4"))
        if max_level < 1:
            return {}

        try:
            coeffs = pywt.wavedec(x, "db4", level=max_level)
        except Exception:
            return self._fallback_multiscale()

        # Energy at each scale
        total_energy = sum(np.sum(c ** 2) for c in coeffs)
        if total_energy == 0:
            return {}

        features = {}
        # Approximation (long-term trend)
        approx_energy = np.sum(coeffs[0] ** 2) / total_energy
        features["wavelet_trend_energy"] = float(approx_energy)

        # Detail coefficients (short-term patterns)
        for i, detail in enumerate(coeffs[1:], 1):
            detail_energy = np.sum(detail ** 2) / total_energy
            features[f"wavelet_detail_{i}_energy"] = float(detail_energy)

        # Trend direction (from approximation coefficients)
        if len(coeffs[0]) >= 2:
            trend_slope = (coeffs[0][-1] - coeffs[0][0]) / len(coeffs[0])
            features["wavelet_trend_direction"] = float(trend_slope)
        else:
            features["wavelet_trend_direction"] = 0.0

        self._wavelet_features = features
        return features

    def _fallback_multiscale(self):
        """Simplified multi-scale analysis when PyWavelets unavailable."""
        x = np.array(self.margins, dtype=float)
        n = len(x)

        features = {}
        # Simple moving average decomposition at different scales
        for scale in [3, 5, 10]:
            if n >= scale:
                smoothed = np.convolve(x, np.ones(scale) / scale, mode="valid")
                detail = x[scale - 1:] - smoothed
                features[f"ma_{scale}_trend"] = float(smoothed[-1] - smoothed[0]) / len(smoothed)
                features[f"ma_{scale}_detail_energy"] = float(np.var(detail))

        self._wavelet_features = features
        return features

    def get_features(self):
        """Get all signal processing features."""
        features = {}
        features.update(self.analyze_fourier())
        features.update(self.analyze_wavelet())
        return features


class LeagueSignalAnalyzer:
    """Manages signal analysis for all teams in a league."""

    def __init__(self, min_games=15):
        self.min_games = min_games
        self.teams = defaultdict(lambda: TeamSignalAnalyzer(min_games))

    def add_game(self, team, margin):
        self.teams[team].add_game(margin)

    def get_features(self, home_team, away_team):
        """Get signal processing features for a matchup."""
        home_feats = self.teams[home_team].get_features() if home_team in self.teams else {}
        away_feats = self.teams[away_team].get_features() if away_team in self.teams else {}

        features = {}

        # Spectral entropy difference (lower = more predictable)
        h_ent = home_feats.get("spectral_entropy", 0.5)
        a_ent = away_feats.get("spectral_entropy", 0.5)
        features["spectral_entropy_diff"] = h_ent - a_ent
        features["combined_entropy"] = (h_ent + a_ent) / 2

        # Trend energy
        h_trend = home_feats.get("wavelet_trend_energy", 0.5)
        a_trend = away_feats.get("wavelet_trend_energy", 0.5)
        features["trend_energy_diff"] = h_trend - a_trend

        # Trend direction
        h_dir = home_feats.get("wavelet_trend_direction", 0)
        a_dir = away_feats.get("wavelet_trend_direction", 0)
        features["trend_direction_diff"] = h_dir - a_dir

        # Dominant period comparison
        h_period = home_feats.get("dominant_period", 10)
        a_period = away_feats.get("dominant_period", 10)
        features["period_diff"] = h_period - a_period

        return features

    def build_from_games(self, games_df, home_col="home_team", away_col="away_team",
                         home_score_col="home_score", away_score_col="away_score"):
        for _, row in games_df.iterrows():
            home = row.get(home_col, "")
            away = row.get(away_col, "")
            try:
                margin = float(row[home_score_col]) - float(row[away_score_col])
            except (ValueError, TypeError):
                continue
            self.add_game(home, margin)
            self.add_game(away, -margin)
        return len(self.teams)
