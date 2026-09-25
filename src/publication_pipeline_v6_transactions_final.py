"""
Rohitash Transactions-Ready Publication Pipeline v6
===========================================

Single-IRS high-speed-railway (HSR) channel estimation and prediction framework.

This file is a self-contained research pipeline designed to address the main
methodological and evaluation issues that commonly arise in IEEE Access review:

1.  Physically consistent single-IRS effective channel H_eff = H_d + H_r Phi G.
2.  Continuous HSR trajectory with Jakes-correlated NLoS evolution.
3.  Incrementally accumulated Doppler phase (not f_D(t)*t shortcut).
4.  Genuine MIMO pilot transmission Y_p = H X_p + N and LS initialization.
5.  Global training normalization (no per-realization normalization).
6.  Proposed spatial linear+nonlinear residual estimator.
7.  CDRN-inspired residual CNN spatial baseline.
8.  CNN-BiLSTM-inspired temporal baseline using the same temporal observation budget.
9.  Empirical matched/mismatched spatial LMMSE baselines.
10. Equal-history empirical temporal LMMSE baseline.
11. Proposed spatial-front-end + condition-aware temporal predictor.
12. Explicit conditioning on speed, SNR, prediction delay, and pilot length.
13. Residual future-channel prediction with a learned gate.
14. Multi-speed / multi-SNR / multi-delay training.
15. Five-seed mean, standard deviation, and 95% confidence intervals.
16. Fair same-trajectory channel-aging evaluation.
17. Jakes-correlation *reference* (not claimed as a strict bound).
18. Genuine pilot-length / overhead study with orthogonal pilot matrices.
19. Achievable-rate evaluation over the TRUE channel using estimated-CSI beams.
20. BER evaluation over the TRUE channel using estimated-CSI beams/equalization.
21. IRS-size scalability with physical (unnormalized) channel-power reporting.
22. Robustness to Rician-K mismatch, unseen speeds, and IRS phase errors.
23. Temporal novelty ablation: speed/SNR/delay/pilot conditioning, gate,
    spatial front end, bidirectionality, and recurrent depth.
24. Complexity: trainable parameters, parameter memory, FLOPs, and latency.
25. Stronger comparison set: LS, Rank-1 SVD baseline, matched/mismatched LMMSE,
    CDRN-inspired baseline, temporal LMMSE, CNN-BiLSTM-inspired baseline,
    GRU-ODE-inspired baseline, GAN-inspired multi-scale baseline, and proposed models.
26. Full numeric export (JSON/CSV) and deterministic experiment metadata.

IMPORTANT SCIENTIFIC FRAMING
----------------------------
- The proposed spatial branch is NOT claimed to guarantee superiority over
  LMMSE. It is designed to combine a learnable linear component with a
  nonlinear residual component.
- The Jakes curve is an analytical correlation reference, NOT a universal
  theoretical lower bound for the Rician direct+cascaded effective channel.
- "CDRN-inspired" and "CNN-BiLSTM-inspired" below are reproducible architecture-level
  baselines, not claims of exact reproduction of any particular paper.
- The main temporal comparison gives temporal baselines the SAME sequence of
  pilot-derived LS estimates as the proposed temporal predictor.

THEORETICAL BASIS
-----------------
The adaptive design is tied to three communication-theoretic relations used
explicitly by the implementation rather than being presented as a collection
of neural blocks:

1. Doppler-conditioned state evolution:
       rho(Delta) = J0(2*pi*f_D*Delta).
   The physics-guided predictor uses this Jakes-correlation-guided evolution,
   together with a deterministic Doppler phase rotation, as the analytic
   prediction term before learning only the residual correction.

2. Orthogonal-pilot LS variance:
       var(H_LS) proportional to 1/(tau*gamma),
   where tau is pilot length and gamma is linear SNR. This motivates the
   uncertainty-aware adaptive pilot policy: shorter pilots are selected when
   uncertainty/noise is low and longer pilots when uncertainty, Doppler, or
   prediction horizon increases.

3. Net-rate tradeoff:
       R_net = (1 - tau/T_c) R.
   The adaptive-pilot objective combines estimation error with the explicit
   tau/T_c overhead term using coherence_symbols().

Together, the closed-loop strategy balances prediction error, pilot overhead,
and channel aging, then uses predicted future CSI to update the IRS phase
configuration. The primary contribution is therefore a mobility-conditioned
predictive effective-channel framework, not a claim of first-ever RIS-HSR
deep-learning channel estimation.

Usage
-----
Publication run (default):
    python publication_pipeline_v6_transactions_final.py --mode publication

Fast implementation smoke test:
    python publication_pipeline_v6_transactions_final.py --mode smoke

Dependencies
------------
    numpy, scipy, pandas, matplotlib, tensorflow
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import j0
from tensorflow.keras import callbacks, layers, models


# =============================================================================
# Reproducibility and numerical helpers
# =============================================================================

SPEED_OF_LIGHT = 299_792_458.0
Array = np.ndarray
ModelInput = Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]


def set_global_seed(seed: int) -> None:
    """Best-effort deterministic setup for NumPy/Python/TensorFlow."""
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 10.0))


def linear_to_db(x: float, floor: float = 1e-30) -> float:
    return float(10.0 * np.log10(max(float(x), floor)))


def dbm_to_watt(dbm: float) -> float:
    return float(10.0 ** ((dbm - 30.0) / 10.0))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def direction_angle(tx: Tuple[float, float], rx: Tuple[float, float]) -> float:
    return float(np.arctan2(rx[1] - tx[1], rx[0] - tx[0]))


def maximum_doppler_hz(speed_mps: float, carrier_hz: float) -> float:
    wavelength = SPEED_OF_LIGHT / carrier_hz
    return float(abs(speed_mps) / wavelength)


def link_doppler_hz(speed_mps: float, carrier_hz: float, propagation_angle_rad: float) -> float:
    # Train velocity is along +x; propagation angle is measured from +x.
    return float(maximum_doppler_hz(speed_mps, carrier_hz) * np.cos(propagation_angle_rad))


def temporal_correlation_jakes(doppler_hz: float, delta_t_s: float) -> float:
    return float(j0(2.0 * np.pi * abs(doppler_hz) * delta_t_s))


def ula_steering_vector(num_elements: int, angle_rad: float, spacing_lambda: float = 0.5) -> Array:
    idx = np.arange(num_elements, dtype=float)
    phase = 2.0 * np.pi * spacing_lambda * idx * np.sin(angle_rad)
    return (np.exp(1j * phase) / np.sqrt(num_elements)).astype(np.complex128)


def complex_gaussian(shape: Tuple[int, ...], rng: np.random.Generator) -> Array:
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)).astype(
        np.complex128
    )


def path_gain_linear(distance_m: float, ref_pathloss_db: float, pathloss_exp: float, ref_distance_m: float) -> float:
    d = max(distance_m, ref_distance_m)
    gain_db = ref_pathloss_db - 10.0 * pathloss_exp * np.log10(d / ref_distance_m)
    return db_to_linear(float(gain_db))


def complex_to_image(h: Array) -> Array:
    return np.stack([h.real, h.imag], axis=-1).astype(np.float32)


def image_to_complex(x: Array) -> Array:
    return x[..., 0].astype(np.float64) + 1j * x[..., 1].astype(np.float64)


def nmse_linear(y_true: Array, y_pred: Array) -> float:
    err = np.sum((y_true - y_pred) ** 2, axis=tuple(range(1, y_true.ndim)))
    sig = np.sum(y_true**2, axis=tuple(range(1, y_true.ndim)))
    return float(np.mean(err / np.maximum(sig, 1e-30)))


def nmse_db(y_true: Array, y_pred: Array) -> float:
    return linear_to_db(nmse_linear(y_true, y_pred))


def summarize_seed_values(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SysConfig:
    num_bs_antennas: int = 8
    num_mcr_antennas: int = 4
    num_irs_elements: int = 32

    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 10e6
    transmit_power_dbm: float = 30.0
    noise_figure_db: float = 7.0
    thermal_noise_density_dbm_hz: float = -174.0

    rician_k_bs_irs_db: float = 18.0
    rician_k_irs_mcr_db: float = 14.0
    rician_k_direct_db: float = 10.0

    pathloss_exp_bs_irs: float = 2.2
    pathloss_exp_irs_mcr: float = 2.2
    pathloss_exp_direct: float = 2.8
    ref_pathloss_db: float = -20.0
    ref_distance_m: float = 1.0
    direct_blockage_db: float = 0.0

    bs_position: Tuple[float, float] = (0.0, 50.0)
    irs_position: Tuple[float, float] = (100.0, 20.0)
    train_x_min_m: float = -150.0
    train_x_max_m: float = 150.0

    # Scenario realism / domain-generalization controls. Defaults preserve the
    # original open-track behavior exactly.
    shadowing_std_db: float = 0.0
    blockage_probability: float = 0.0
    blockage_extra_db: float = 0.0

    # Wideband MIMO-OFDM extension. The original narrowband effective-channel
    # functions remain available; dedicated *_ofdm helpers add frequency
    # selectivity without silently changing the original API.
    num_subcarriers: int = 64
    subcarrier_spacing_hz: float = 120_000.0
    num_channel_taps: int = 4
    max_excess_delay_s: float = 0.8e-6
    ofdm_symbol_duration_s: float = 1.0 / 120_000.0

    @classmethod
    def open_track_viaduct(cls) -> "SysConfig":
        """Strong-LoS open-track/viaduct preset."""
        return cls(
            rician_k_bs_irs_db=18.0,
            rician_k_irs_mcr_db=14.0,
            rician_k_direct_db=10.0,
            pathloss_exp_bs_irs=2.2,
            pathloss_exp_irs_mcr=2.2,
            pathloss_exp_direct=2.8,
            shadowing_std_db=2.0,
            blockage_probability=0.0,
        )

    @classmethod
    def tunnel_cutting(cls) -> "SysConfig":
        """Lower-K, stronger-multipath tunnel/cutting preset."""
        return cls(
            rician_k_bs_irs_db=10.0,
            rician_k_irs_mcr_db=5.0,
            rician_k_direct_db=2.0,
            pathloss_exp_bs_irs=2.5,
            pathloss_exp_irs_mcr=2.8,
            pathloss_exp_direct=3.4,
            shadowing_std_db=6.0,
            blockage_probability=0.15,
            blockage_extra_db=18.0,
        )


@dataclass
class RunConfig:
    output_dir: str = "rohitash_ieee_access_results"

    # Main training/evaluation
    n_seeds: int = 5
    num_train_spatial: int = 40_000
    num_val_spatial: int = 5_000
    num_train_temporal: int = 20_000
    num_val_temporal: int = 2_500
    num_test: int = 2_000
    covariance_samples: int = 3_000
    epochs_spatial: int = 100
    epochs_temporal: int = 100
    batch_size_spatial: int = 128
    batch_size_temporal: int = 64
    learning_rate_spatial: float = 1e-3
    learning_rate_temporal: float = 1e-3

    # Normalization is global and fixed for all models/seeds.
    normalization_seed: int = 20260808
    normalization_samples: int = 6_000

    # Training ranges. Highest speed and longest delay are intentionally held out.
    snr_train_min_db: float = -10.0
    snr_train_max_db: float = 30.0
    speed_train_min_kmh: float = 0.0
    speed_train_max_kmh: float = 400.0
    delay_train_min_ms: float = 0.0
    delay_train_max_ms: float = 3.0
    training_pilot_lengths: Tuple[int, ...] = (8, 12, 16, 24)

    # Temporal observation structure
    seq_len: int = 4
    history_dt_s: float = 0.3e-3
    main_pilot_length: int = 8

    # Evaluation grids
    snr_points_db: Tuple[float, ...] = (-10, -5, 0, 5, 10, 15, 20, 25, 30)
    speed_points_kmh: Tuple[float, ...] = (0, 100, 200, 300, 400, 450, 500)
    aging_delays_ms: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0)
    pilot_lengths: Tuple[int, ...] = (8, 12, 16, 24, 32)
    irs_sizes: Tuple[int, ...] = (16, 32, 64, 128)
    k_offsets_db: Tuple[float, ...] = (0.0, -5.0, -10.0, -15.0)
    phase_error_std_deg: Tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)

    fixed_eval_snr_db: float = 15.0
    fixed_eval_speed_kmh: float = 300.0
    speed_eval_prediction_delay_ms: float = 1.0
    pilot_eval_prediction_delay_ms: float = 0.5

    # V6: the genuine (nonzero) prediction horizon used for the headline
    # NMSE-vs-SNR and BER/SE-vs-SNR figures, so the proposed model's
    # predictive claim is evaluated where it actually has to predict, not
    # only at delay=0 where it can default to near-identity.
    headline_prediction_delay_ms: float = 1.0

    # Practical LMMSE mismatch used in main SNR plot.
    lmmse_mismatch_k_offset_db: float = -10.0

    # Robustness/ablation workload
    ablation_seeds: int = 5
    ablation_train_samples: int = 12_000
    ablation_val_samples: int = 1_500
    ablation_epochs: int = 60

    irs_sweep_seeds: int = 3
    irs_sweep_train_samples: int = 12_000
    irs_sweep_val_samples: int = 1_500
    irs_sweep_epochs: int = 60
    irs_sweep_test_samples: int = 800

    # BER/SE workloads
    ber_channel_samples: int = 500
    ber_symbols_per_channel: int = 2_000
    rate_channel_samples: int = 500

    # Optional end-to-end fine tuning of the proposed temporal model.
    temporal_finetune_epochs: int = 10

    # V6 probabilistic-training stabilization. NLL and MSE have different
    # optimization scales, so the uncertainty-aware model has its own LR and
    # a conservative initial log variance. Smoke mode intentionally keeps the
    # same one-epoch structural test; publication mode uses the full budget.
    learning_rate_probabilistic: float = 5e-4
    initial_log_variance: float = -2.0


    # Priority-0 closed-loop controls.
    use_physics_guided_proposed: bool = True
    use_uncertainty_nll: bool = True
    uncertainty_bins: int = 10
    adaptive_pilot_lambda: float = 0.05
    adaptive_policy_trajectories: int = 64
    adaptive_policy_steps: int = 24
    adaptive_allow_unseen_pilot_lengths: bool = False
    irs_control_trajectories: int = 64
    irs_phase_iterations: int = 4

    # Priority-1 baseline switches. MultiScale is deliberately described as
    # GAN-inspired/non-adversarial unless an exact adversarial reproduction is
    # implemented and validated separately.
    include_gan_baseline: bool = True
    include_ode_baseline: bool = True

    # Wideband evaluation workload. Shared estimator weights are applied per
    # subcarrier; a dedicated cross-frequency network is intentionally not
    # claimed.
    include_ofdm_extension: bool = True
    ofdm_test_samples: int = 96
    ofdm_pilot_subcarrier_stride: int = 4

    # Cross-scenario/domain-generalization workload.
    cross_scenario_seeds: int = 5
    cross_scenario_train_samples: int = 8_000
    cross_scenario_val_samples: int = 1_000
    cross_scenario_epochs: int = 50

    @staticmethod
    def smoke() -> "RunConfig":
        return RunConfig(
            output_dir="rohitash_ieee_access_smoke",
            n_seeds=1,
            num_train_spatial=128,
            num_val_spatial=32,
            num_train_temporal=96,
            num_val_temporal=24,
            num_test=48,
            covariance_samples=96,
            epochs_spatial=1,
            epochs_temporal=1,
            batch_size_spatial=32,
            batch_size_temporal=16,
            normalization_samples=128,
            snr_points_db=(0, 15),
            speed_points_kmh=(100, 500),
            aging_delays_ms=(0.0, 1.0),
            pilot_lengths=(8, 16),
            irs_sizes=(16,),
            k_offsets_db=(0.0, -10.0),
            phase_error_std_deg=(0.0, 10.0),
            ablation_seeds=1,
            ablation_train_samples=64,
            ablation_val_samples=16,
            ablation_epochs=1,
            irs_sweep_seeds=1,
            irs_sweep_train_samples=64,
            irs_sweep_val_samples=16,
            irs_sweep_epochs=1,
            irs_sweep_test_samples=32,
            ber_channel_samples=16,
            ber_symbols_per_channel=64,
            rate_channel_samples=24,
            temporal_finetune_epochs=0,
            adaptive_policy_trajectories=2,
            adaptive_policy_steps=6,
            irs_control_trajectories=2,
            include_gan_baseline=True,
            include_ode_baseline=True,
            include_ofdm_extension=True,
            ofdm_test_samples=2,
            cross_scenario_seeds=1,
            cross_scenario_train_samples=64,
            cross_scenario_val_samples=16,
            cross_scenario_epochs=1,
        )


# =============================================================================
# Physical Rician channel and continuous HSR trajectory
# =============================================================================


def rician_channel_snapshot(
    num_rx: int,
    num_tx: int,
    distance_m: float,
    angle_departure_rad: float,
    angle_arrival_rad: float,
    rician_k_db: float,
    pathloss_exponent: float,
    config: SysConfig,
    rng: np.random.Generator,
    nlos_state: Optional[Array] = None,
    temporal_correlation: float = 0.0,
    doppler_phase_rad: float = 0.0,
) -> Tuple[Array, Array]:
    k_linear = db_to_linear(rician_k_db)
    beta = path_gain_linear(
        distance_m,
        config.ref_pathloss_db,
        pathloss_exponent,
        config.ref_distance_m,
    )

    a_rx = ula_steering_vector(num_rx, angle_arrival_rad)
    a_tx = ula_steering_vector(num_tx, angle_departure_rad)
    h_los = (
        np.sqrt(num_rx * num_tx)
        * np.outer(a_rx, np.conjugate(a_tx))
        * np.exp(1j * doppler_phase_rad)
    )

    if nlos_state is None:
        updated_nlos = complex_gaussian((num_rx, num_tx), rng)
    else:
        rho = float(np.clip(temporal_correlation, -0.999999, 0.999999))
        innovation = complex_gaussian((num_rx, num_tx), rng)
        updated_nlos = rho * nlos_state + np.sqrt(max(0.0, 1.0 - rho**2)) * innovation

    h = np.sqrt(beta) * (
        np.sqrt(k_linear / (k_linear + 1.0)) * h_los
        + np.sqrt(1.0 / (k_linear + 1.0)) * updated_nlos
    )
    return h.astype(np.complex128), updated_nlos.astype(np.complex128)


def _perturbed_k(base_k_db: float, offset_db: float) -> float:
    # Negative K values are mathematically valid in dB, but we cap extremely
    # diffuse cases to keep the simulation numerically well behaved.
    return float(max(-20.0, base_k_db + offset_db))


def generate_effective_channel_trajectory(
    cfg: SysConfig,
    rng: np.random.Generator,
    speed_mps: float,
    times_s: Sequence[float],
    *,
    k_offset_db: float = 0.0,
    phase_error_std_deg: float = 0.0,
    start_x_m: Optional[float] = None,
) -> Array:
    """Generate one physically consistent H_eff trajectory at requested times.

    The BS-IRS channel and nominal IRS phase vector are fixed for the complete
    trajectory. Train geometry evolves continuously. Direct and IRS-MCR NLoS
    states evolve recursively using Jakes correlation. Doppler phase is
    accumulated incrementally using phi[k]=phi[k-1]+2*pi*f_D[k]*dt.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times_s must be a non-empty one-dimensional sequence")
    if np.any(np.diff(times) < -1e-15):
        raise ValueError("times_s must be nondecreasing")

    x0 = (
        float(start_x_m)
        if start_x_m is not None
        else float(rng.uniform(cfg.train_x_min_m, cfg.train_x_max_m))
    )

    # Per-trajectory large-scale shadowing. Defaults (0 dB std) reduce to 1.0
    # and therefore preserve the original behavior. The direct and reflected
    # portions use independent shadowing factors to avoid an unrealistically
    # common multiplicative fade.
    shadow_d_db = float(rng.normal(0.0, cfg.shadowing_std_db)) if cfg.shadowing_std_db > 0 else 0.0
    shadow_r_db = float(rng.normal(0.0, cfg.shadowing_std_db)) if cfg.shadowing_std_db > 0 else 0.0
    shadow_d_amp = np.sqrt(db_to_linear(shadow_d_db))
    shadow_r_amp = np.sqrt(db_to_linear(shadow_r_db))
    trajectory_blocked = bool(
        cfg.blockage_probability > 0.0 and rng.random() < cfg.blockage_probability
    )

    # Quasi-static BS -> IRS link.
    d_bi = distance(cfg.bs_position, cfg.irs_position)
    a_bi_dep = direction_angle(cfg.bs_position, cfg.irs_position)
    a_bi_arr = direction_angle(cfg.irs_position, cfg.bs_position)
    g_bs_irs, _ = rician_channel_snapshot(
        cfg.num_irs_elements,
        cfg.num_bs_antennas,
        d_bi,
        a_bi_dep,
        a_bi_arr,
        _perturbed_k(cfg.rician_k_bs_irs_db, k_offset_db),
        cfg.pathloss_exp_bs_irs,
        cfg,
        rng,
    )

    phi = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, cfg.num_irs_elements))
    if phase_error_std_deg > 0.0:
        phase_err = np.deg2rad(phase_error_std_deg) * rng.standard_normal(cfg.num_irs_elements)
        phi = phi * np.exp(1j * phase_err)
    cascaded_fixed = phi[:, None] * g_bs_irs

    nlos_d: Optional[Array] = None
    nlos_r: Optional[Array] = None
    phase_d = 0.0
    phase_r = 0.0
    previous_time = float(times[0])

    out = np.zeros(
        (len(times), cfg.num_mcr_antennas, cfg.num_bs_antennas), dtype=np.complex128
    )

    for idx, t in enumerate(times):
        dt = 0.0 if idx == 0 else float(t - previous_time)
        train_pos = (x0 + speed_mps * float(t), 0.0)

        # Direct BS -> MCR.
        d_bd = distance(cfg.bs_position, train_pos)
        a_bd_dep = direction_angle(cfg.bs_position, train_pos)
        a_bd_arr = direction_angle(train_pos, cfg.bs_position)
        fd_d = link_doppler_hz(speed_mps, cfg.carrier_frequency_hz, a_bd_dep)
        rho_d = temporal_correlation_jakes(fd_d, dt) if idx > 0 else 0.0
        if idx > 0:
            phase_d += 2.0 * np.pi * fd_d * dt

        h_d, nlos_d = rician_channel_snapshot(
            cfg.num_mcr_antennas,
            cfg.num_bs_antennas,
            d_bd,
            a_bd_dep,
            a_bd_arr,
            _perturbed_k(cfg.rician_k_direct_db, k_offset_db),
            cfg.pathloss_exp_direct,
            cfg,
            rng,
            nlos_state=nlos_d,
            temporal_correlation=rho_d,
            doppler_phase_rad=phase_d,
        )
        h_d = h_d * shadow_d_amp
        total_block_db = cfg.direct_blockage_db + (cfg.blockage_extra_db if trajectory_blocked else 0.0)
        if total_block_db > 0:
            h_d = h_d * np.sqrt(db_to_linear(-total_block_db))

        # IRS -> MCR.
        d_rm = distance(cfg.irs_position, train_pos)
        a_rm_dep = direction_angle(cfg.irs_position, train_pos)
        a_rm_arr = direction_angle(train_pos, cfg.irs_position)
        fd_r = link_doppler_hz(speed_mps, cfg.carrier_frequency_hz, a_rm_dep)
        rho_r = temporal_correlation_jakes(fd_r, dt) if idx > 0 else 0.0
        if idx > 0:
            phase_r += 2.0 * np.pi * fd_r * dt

        h_r, nlos_r = rician_channel_snapshot(
            cfg.num_mcr_antennas,
            cfg.num_irs_elements,
            d_rm,
            a_rm_dep,
            a_rm_arr,
            _perturbed_k(cfg.rician_k_irs_mcr_db, k_offset_db),
            cfg.pathloss_exp_irs_mcr,
            cfg,
            rng,
            nlos_state=nlos_r,
            temporal_correlation=rho_r,
            doppler_phase_rad=phase_r,
        )

        h_reflected = (h_r @ cascaded_fixed) * shadow_r_amp
        out[idx] = h_d + h_reflected
        previous_time = float(t)

    return out


# =============================================================================
# Genuine pilot observation model and LS estimate
# =============================================================================


def dft_pilot_matrix(num_tx: int, pilot_length: int) -> Array:
    """Orthogonal row pilot matrix X with X X^H = tau I."""
    if pilot_length < num_tx:
        raise ValueError(
            f"pilot_length={pilot_length} must be >= num_bs_antennas={num_tx} "
            "for full-rank orthogonal LS estimation"
        )
    k = np.arange(num_tx, dtype=float)[:, None]
    n = np.arange(pilot_length, dtype=float)[None, :]
    x = np.exp(-1j * 2.0 * np.pi * k * n / pilot_length)
    return x.astype(np.complex128)


def pilot_observation_and_ls(
    h_true: Array,
    snr_db: float,
    pilot_length: int,
    rng: np.random.Generator,
) -> Tuple[Array, Array, float]:
    """Transmit orthogonal pilots through H and return (Y_p, H_LS, noise_var).

    The requested SNR is defined at the pilot observation Y=H X+N before LS.
    Increasing tau therefore decreases the LS estimation variance naturally.
    """
    x_p = dft_pilot_matrix(h_true.shape[1], pilot_length)
    clean_y = h_true @ x_p
    signal_power = float(np.mean(np.abs(clean_y) ** 2))
    noise_var = signal_power / db_to_linear(snr_db)
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(clean_y.shape) + 1j * rng.standard_normal(clean_y.shape)
    )
    y_p = clean_y + noise
    h_ls = (y_p @ np.conjugate(x_p.T)) / float(pilot_length)
    return y_p.astype(np.complex128), h_ls.astype(np.complex128), float(noise_var)


# =============================================================================
# Global normalization
# =============================================================================


def compute_global_channel_scale(cfg: SysConfig, run: RunConfig) -> float:
    rng = np.random.default_rng(run.normalization_seed)
    powers: List[float] = []
    for _ in range(run.normalization_samples):
        speed_kmh = rng.uniform(run.speed_train_min_kmh, run.speed_train_max_kmh)
        h = generate_effective_channel_trajectory(
            cfg, rng, speed_kmh / 3.6, [0.0]
        )[0]
        powers.append(float(np.mean(np.abs(h) ** 2)))
    scale = float(np.sqrt(np.mean(powers)))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Failed to compute a valid global channel normalization scale")
    return scale


# =============================================================================
# Dataset builders
# =============================================================================


def _snr_sample_weight(snr_db: float) -> float:
    # Mild weighting preserves high-SNR identity behavior without allowing the
    # high-SNR end of the training range to dominate the entire objective.
    return float(np.clip(10.0 ** (snr_db / 20.0), 0.25, 12.0))


def build_spatial_dataset(
    cfg: SysConfig,
    run: RunConfig,
    n: int,
    rng: np.random.Generator,
    scale: float,
    *,
    fixed_snr_db: Optional[float] = None,
    fixed_speed_kmh: Optional[float] = None,
    fixed_pilot_length: Optional[int] = None,
    k_offset_db: float = 0.0,
    phase_error_std_deg: float = 0.0,
    return_raw: bool = False,
) -> Tuple[Array, Array, Array, Optional[Dict[str, Array]]]:
    x = np.zeros((n, cfg.num_mcr_antennas, cfg.num_bs_antennas, 2), dtype=np.float32)
    y = np.zeros_like(x)
    w = np.zeros(n, dtype=np.float32)

    raw_true = np.zeros((n, cfg.num_mcr_antennas, cfg.num_bs_antennas), dtype=np.complex128)
    raw_ls = np.zeros_like(raw_true)
    snrs = np.zeros(n, dtype=np.float32)
    speeds = np.zeros(n, dtype=np.float32)
    taus = np.zeros(n, dtype=np.int32)

    for i in range(n):
        snr = (
            float(fixed_snr_db)
            if fixed_snr_db is not None
            else float(rng.uniform(run.snr_train_min_db, run.snr_train_max_db))
        )
        speed = (
            float(fixed_speed_kmh)
            if fixed_speed_kmh is not None
            else float(rng.uniform(run.speed_train_min_kmh, run.speed_train_max_kmh))
        )
        tau = (
            int(fixed_pilot_length)
            if fixed_pilot_length is not None
            else int(rng.choice(run.training_pilot_lengths))
        )

        h_true = generate_effective_channel_trajectory(
            cfg,
            rng,
            speed / 3.6,
            [0.0],
            k_offset_db=k_offset_db,
            phase_error_std_deg=phase_error_std_deg,
        )[0]
        _, h_ls, _ = pilot_observation_and_ls(h_true, snr, tau, rng)

        x[i] = complex_to_image(h_ls / scale)
        y[i] = complex_to_image(h_true / scale)
        w[i] = _snr_sample_weight(snr)
        raw_true[i] = h_true
        raw_ls[i] = h_ls
        snrs[i], speeds[i], taus[i] = snr, speed, tau

    w /= max(float(np.mean(w)), 1e-12)
    metadata = None
    if return_raw:
        metadata = {
            "raw_true": raw_true,
            "raw_ls": raw_ls,
            "snr_db": snrs,
            "speed_kmh": speeds,
            "pilot_length": taus,
        }
    return x, y, w, metadata


def _condition_arrays(
    n: int,
    speed_kmh: Array,
    snr_db: Array,
    delay_ms: Array,
    pilot_length: Array,
    run: RunConfig,
) -> List[Array]:
    speed_norm = (speed_kmh / 500.0).reshape(n, 1).astype(np.float32)
    snr_norm = ((snr_db + 10.0) / 40.0).reshape(n, 1).astype(np.float32)
    delay_norm = (delay_ms / max(run.delay_train_max_ms, 1e-6)).reshape(n, 1).astype(np.float32)
    pilot_norm = (pilot_length / max(run.pilot_lengths)).reshape(n, 1).astype(np.float32)
    return [speed_norm, snr_norm, delay_norm, pilot_norm]


def build_temporal_dataset(
    cfg: SysConfig,
    run: RunConfig,
    n: int,
    rng: np.random.Generator,
    scale: float,
    *,
    fixed_snr_db: Optional[float] = None,
    fixed_speed_kmh: Optional[float] = None,
    fixed_delay_ms: Optional[float] = None,
    fixed_pilot_length: Optional[int] = None,
    k_offset_db: float = 0.0,
    phase_error_std_deg: float = 0.0,
    return_raw: bool = False,
) -> Tuple[List[Array], Array, Array, Optional[Dict[str, Array]]]:
    n_rx, n_tx = cfg.num_mcr_antennas, cfg.num_bs_antennas
    seq = np.zeros((n, run.seq_len, n_rx, n_tx, 2), dtype=np.float32)
    target = np.zeros((n, n_rx, n_tx, 2), dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)

    snrs = np.zeros(n, dtype=np.float32)
    speeds = np.zeros(n, dtype=np.float32)
    delays = np.zeros(n, dtype=np.float32)
    taus = np.zeros(n, dtype=np.float32)

    raw_hist = np.zeros((n, run.seq_len, n_rx, n_tx), dtype=np.complex128)
    raw_ls_hist = np.zeros_like(raw_hist)
    raw_target = np.zeros((n, n_rx, n_tx), dtype=np.complex128)

    history_times = np.arange(run.seq_len, dtype=float) * run.history_dt_s
    current_time = float(history_times[-1])

    for i in range(n):
        snr = (
            float(fixed_snr_db)
            if fixed_snr_db is not None
            else float(rng.uniform(run.snr_train_min_db, run.snr_train_max_db))
        )
        speed = (
            float(fixed_speed_kmh)
            if fixed_speed_kmh is not None
            else float(rng.uniform(run.speed_train_min_kmh, run.speed_train_max_kmh))
        )
        delay = (
            float(fixed_delay_ms)
            if fixed_delay_ms is not None
            else float(rng.uniform(run.delay_train_min_ms, run.delay_train_max_ms))
        )
        tau = (
            int(fixed_pilot_length)
            if fixed_pilot_length is not None
            else int(rng.choice(run.training_pilot_lengths))
        )

        target_time = current_time + delay / 1000.0
        requested_times = list(history_times)
        if target_time > current_time + 1e-15:
            requested_times.append(target_time)

        h_all = generate_effective_channel_trajectory(
            cfg,
            rng,
            speed / 3.6,
            requested_times,
            k_offset_db=k_offset_db,
            phase_error_std_deg=phase_error_std_deg,
        )
        h_history = h_all[: run.seq_len]
        h_targ = h_history[-1] if len(h_all) == run.seq_len else h_all[-1]

        for t in range(run.seq_len):
            _, h_ls, _ = pilot_observation_and_ls(h_history[t], snr, tau, rng)
            seq[i, t] = complex_to_image(h_ls / scale)
            raw_hist[i, t] = h_history[t]
            raw_ls_hist[i, t] = h_ls

        target[i] = complex_to_image(h_targ / scale)
        raw_target[i] = h_targ
        weights[i] = _snr_sample_weight(snr)
        snrs[i], speeds[i], delays[i], taus[i] = snr, speed, delay, tau

    weights /= max(float(np.mean(weights)), 1e-12)
    cond = _condition_arrays(n, speeds, snrs, delays, taus, run)
    model_inputs = [seq] + cond

    metadata = None
    if return_raw:
        metadata = {
            "raw_history": raw_hist,
            "raw_ls_history": raw_ls_hist,
            "raw_target": raw_target,
            "snr_db": snrs,
            "speed_kmh": speeds,
            "delay_ms": delays,
            "pilot_length": taus,
        }
    return model_inputs, target, weights, metadata


# =============================================================================
# Neural models
# =============================================================================


@tf.keras.utils.register_keras_serializable(package="rohitash_custom_losses")
def per_sample_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    axes = list(range(1, len(y_true.shape)))
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=axes)


def build_proposed_spatial(
    input_shape: Tuple[int, int, int],
    hidden_units: int = 256,
    residual_blocks: int = 3,
    ablation: str = "full",
) -> tf.keras.Model:
    """Linear + nonlinear residual channel estimator.

    The linear branch can learn a data-driven linear correction; the nonlinear
    residual branch models dependencies beyond a purely linear estimator.
    No claim of guaranteed LMMSE equivalence/superiority is made.
    """
    inputs = layers.Input(shape=input_shape, name="ls_channel")
    feat_dim = int(np.prod(input_shape))
    flat = layers.Flatten()(inputs)

    linear_noise = layers.Dense(feat_dim, use_bias=False, name="linear_noise")(flat)

    z = layers.Dense(hidden_units, activation="relu", name="nl_in")(flat)
    for b in range(residual_blocks):
        r = layers.Dense(hidden_units, activation="relu", name=f"nl_b{b}_1")(z)
        r = layers.Dense(hidden_units, activation="relu", name=f"nl_b{b}_2")(r)
        z = layers.Add(name=f"nl_b{b}_add")([z, r])
    nonlinear_noise = layers.Dense(feat_dim, name="nonlinear_noise")(z)

    if ablation == "linear_only":
        noise_flat = linear_noise
    elif ablation == "nonlinear_only":
        noise_flat = nonlinear_noise
    else:
        noise_flat = layers.Add(name="hybrid_noise")([linear_noise, nonlinear_noise])

    noise = layers.Reshape(input_shape)(noise_flat)
    outputs = layers.Subtract(name="residual_cleaning")([inputs, noise])
    return models.Model(inputs, outputs, name="PropSpatial")


def build_cdrn_baseline(input_shape: Tuple[int, int, int], depth: int = 8, filters: int = 64) -> tf.keras.Model:
    """CDRN-inspired residual convolutional denoiser baseline."""
    inputs = layers.Input(shape=input_shape, name="ls_channel")
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    for _ in range(depth - 2):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    noise = layers.Conv2D(input_shape[-1], 3, padding="same", activation="linear")(x)
    outputs = layers.Subtract()([inputs, noise])
    return models.Model(inputs, outputs, name="CDRNInspired")


def build_cnn_bilstm_baseline(seq_len: int, n_rx: int, n_tx: int) -> tf.keras.Model:
    """Strong conditioned CNN-BiLSTM-inspired temporal baseline.

    It receives the same temporal pilot-LS sequence and the same side
    information (speed, SNR, prediction delay, pilot length) as the proposed
    predictor. This avoids handicapping the baseline simply because the
    proposed model has access to metadata. The proposed model must therefore
    earn its gain from the shared spatial front end, gated residual prediction,
    and its joint architecture rather than from extra inputs alone.
    """
    seq_in = layers.Input(shape=(seq_len, n_rx, n_tx, 2), name="pilot_ls_seq")
    speed_in = layers.Input(shape=(1,), name="speed_norm")
    snr_in = layers.Input(shape=(1,), name="snr_norm")
    delay_in = layers.Input(shape=(1,), name="delay_norm")
    pilot_in = layers.Input(shape=(1,), name="pilot_norm")

    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(seq_in)
    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=False))(x)

    cond = layers.Concatenate()([speed_in, snr_in, delay_in, pilot_in])
    cond = layers.Dense(32, activation="relu")(cond)
    x = layers.Concatenate()([x, cond])
    x = layers.Dense(128, activation="relu")(x)
    delta = layers.Dense(n_rx * n_tx * 2, kernel_initializer="zeros", bias_initializer="zeros")(x)
    delta = layers.Reshape((n_rx, n_tx, 2))(delta)
    last = layers.Lambda(lambda t: t[:, -1], name="last_ls")(seq_in)
    outputs = layers.Add(name="future_residual")([last, delta])
    return models.Model(
        [seq_in, speed_in, snr_in, delay_in, pilot_in],
        outputs,
        name="Conditioned_CNN_BiLSTM",
    )


def _masked_scalar_input(inp: tf.Tensor, enabled: bool, name: str) -> tf.Tensor:
    if enabled:
        return inp
    return layers.Lambda(lambda x: tf.zeros_like(x), name=f"zero_{name}")(inp)


def build_proposed_temporal(
    spatial_frontend: tf.keras.Model,
    seq_len: int,
    n_rx: int,
    n_tx: int,
    *,
    use_speed: bool = True,
    use_snr: bool = True,
    use_delay: bool = True,
    use_pilot: bool = True,
    use_spatial_frontend: bool = True,
    use_gate: bool = True,
    bidirectional: bool = True,
    recurrent_layers: int = 2,
) -> tf.keras.Model:
    """Condition-aware residual future-channel predictor.

    Pipeline:
      pilot-derived LS sequence
        -> shared PropSpatial front end
        -> temporal encoder
        -> condition fusion [speed, SNR, delay, pilot length]
        -> gated residual future-channel correction.
    """
    seq_in = layers.Input(shape=(seq_len, n_rx, n_tx, 2), name="pilot_ls_seq")
    speed_in = layers.Input(shape=(1,), name="speed_norm")
    snr_in = layers.Input(shape=(1,), name="snr_norm")
    delay_in = layers.Input(shape=(1,), name="delay_norm")
    pilot_in = layers.Input(shape=(1,), name="pilot_norm")

    if use_spatial_frontend:
        frames = layers.TimeDistributed(spatial_frontend, name="shared_spatial_frontend")(seq_in)
    else:
        frames = seq_in

    feat_dim = n_rx * n_tx * 2
    x = layers.TimeDistributed(layers.Flatten())(frames)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)

    cond_parts = [
        _masked_scalar_input(speed_in, use_speed, "speed"),
        _masked_scalar_input(snr_in, use_snr, "snr"),
        _masked_scalar_input(delay_in, use_delay, "delay"),
        _masked_scalar_input(pilot_in, use_pilot, "pilot"),
    ]
    cond = layers.Concatenate(name="condition_vector")(cond_parts)
    cond = layers.Dense(64, activation="relu", name="condition_embedding")(cond)
    cond_seq = layers.RepeatVector(seq_len)(cond)
    x = layers.Concatenate(axis=-1)([x, cond_seq])

    rnn_units = 128
    for layer_idx in range(max(1, recurrent_layers)):
        return_seq = layer_idx < max(1, recurrent_layers) - 1
        base = layers.LSTM(rnn_units, return_sequences=return_seq, name=f"lstm_{layer_idx}")
        x = layers.Bidirectional(base, name=f"bilstm_{layer_idx}")(x) if bidirectional else base(x)

    x = layers.Concatenate()([x, cond])
    x = layers.Dense(256, activation="relu")(x)
    delta = layers.Dense(feat_dim, kernel_initializer="zeros", bias_initializer="zeros", name="future_delta")(x)

    if use_gate:
        gate = layers.Dense(feat_dim, activation="sigmoid", name="residual_gate")(x)
        delta = layers.Multiply()([delta, gate])

    delta = layers.Reshape((n_rx, n_tx, 2))(delta)
    last_frame = layers.Lambda(lambda t: t[:, -1], name="last_spatial_frame")(frames)
    outputs = layers.Add(name="predicted_future_channel")([last_frame, delta])

    return models.Model(
        [seq_in, speed_in, snr_in, delay_in, pilot_in], outputs, name="ConditionedTemporalPredictor"
    )



# =============================================================================
# Physics-guided predictive model, uncertainty, and stronger baselines
# =============================================================================


@tf.keras.utils.register_keras_serializable(package="rohitash_custom_losses")
def heteroscedastic_nll_loss(y_true: tf.Tensor, y_pred_packed: tf.Tensor) -> tf.Tensor:
    """Element-wise Gaussian negative log-likelihood for real/imag channels.

    The model packs [mu_real, mu_imag, logvar_real, logvar_imag] along the
    final axis. Clipping log-variance prevents numerical collapse while still
    allowing the model to express several orders of magnitude of uncertainty.
    """
    channels = tf.shape(y_true)[-1]
    mu = y_pred_packed[..., :channels]
    log_sigma2 = tf.clip_by_value(y_pred_packed[..., channels:], -4.0, 8.0)
    inv_var = tf.exp(-log_sigma2)
    nll = tf.square(y_true - mu) * inv_var + log_sigma2
    axes = list(range(1, len(y_true.shape)))
    return tf.reduce_mean(nll, axis=axes)


@tf.keras.utils.register_keras_serializable(package="rohitash_custom_layers")
class PhysicsGuidedJakesLayer(layers.Layer):
    """Jakes-correlation-guided effective-channel prediction prior.

    V6 avoids imposing one always-positive maximum-Doppler rotation on the
    complete direct+IRS effective channel. Instead, it estimates the signed
    *effective* phase increment from the two most recent spatial channel
    estimates, converts that increment to an effective Doppler frequency, and
    clips the estimate to the physically admissible speed-derived maximum
    Doppler. The requested prediction horizon then determines the deterministic
    phase extrapolation and J0 correlation magnitude.

    This remains an analytic prior rather than a claim that one scalar Doppler
    exactly describes every constituent propagation path. The learned residual
    branch corrects the remaining multi-link/model mismatch.
    """

    def __init__(
        self,
        carrier_frequency_hz: float,
        history_dt_s: float,
        speed_norm_kmh: float = 500.0,
        delay_norm_ms: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.history_dt_s = float(history_dt_s)
        self.speed_norm_kmh = float(speed_norm_kmh)
        self.delay_norm_ms = float(delay_norm_ms)


    def call(self, inputs):
        prev_frame, last_frame, speed_norm, delay_norm = inputs
        dtype = last_frame.dtype
        speed_mps = speed_norm[:, 0] * tf.cast(self.speed_norm_kmh / 3.6, dtype)
        delay_s = delay_norm[:, 0] * tf.cast(self.delay_norm_ms / 1000.0, dtype)
        wavelength = tf.cast(SPEED_OF_LIGHT / self.carrier_frequency_hz, dtype)
        max_fd = tf.abs(speed_mps) / wavelength

        # Effective signed phase progression from conj(H_{t-1}) * H_t.
        pr, pi = prev_frame[..., 0], prev_frame[..., 1]
        cr, ci = last_frame[..., 0], last_frame[..., 1]
        cross_re = tf.reduce_sum(pr * cr + pi * ci, axis=[1, 2])
        cross_im = tf.reduce_sum(pr * ci - pi * cr, axis=[1, 2])
        phase_step = tf.atan2(cross_im, cross_re)

        hist_dt = tf.cast(max(self.history_dt_s, 1e-9), dtype)
        fd_observed = phase_step / (tf.cast(2.0 * np.pi, dtype) * hist_dt)
        fd_effective = tf.clip_by_value(fd_observed, -max_fd, max_fd)

        x = tf.cast(2.0 * np.pi, dtype) * tf.abs(fd_effective) * delay_s
        rho = tf.numpy_function(lambda z: j0(z).astype(np.float32), [tf.cast(x, tf.float32)], tf.float32)
        rho.set_shape(x.shape)
        rho = tf.cast(rho, dtype)

        phase = tf.cast(2.0 * np.pi, dtype) * fd_effective * delay_s
        c, s = tf.cos(phase), tf.sin(phase)
        c = c[:, None, None]
        s = s[:, None, None]
        rho = rho[:, None, None]

        re = last_frame[..., 0]
        im = last_frame[..., 1]
        re2 = rho * (re * c - im * s)
        im2 = rho * (re * s + im * c)
        return tf.stack([re2, im2], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "carrier_frequency_hz": self.carrier_frequency_hz,
                "history_dt_s": self.history_dt_s,
                "speed_norm_kmh": self.speed_norm_kmh,
                "delay_norm_ms": self.delay_norm_ms,
            }
        )
        return cfg



def build_proposed_temporal_physics_guided(
    spatial_frontend: tf.keras.Model,
    seq_len: int,
    n_rx: int,
    n_tx: int,
    *,
    carrier_frequency_hz: float,
    delay_norm_ms: float,
    history_dt_s: float,
    use_speed: bool = True,
    use_snr: bool = True,
    use_delay: bool = True,
    use_pilot: bool = True,
    use_spatial_frontend: bool = True,
    use_gate: bool = True,
    use_physics_prior: bool = True,
    bidirectional: bool = True,
    recurrent_layers: int = 2,
    predict_uncertainty: bool = True,
    initial_log_variance: float = -2.0,
) -> tf.keras.Model:
    """Physics-guided, uncertainty-aware future effective-channel predictor.

    The deterministic branch extrapolates the latest channel using a signed
    effective Doppler inferred from consecutive channel frames and bounded by
    the known speed-derived maximum Doppler. The recurrent network predicts
    only the residual correction. The uncertainty head predicts element-wise
    log variance.

    Ablations are structurally clean in V6: disabling speed or delay removes
    that quantity from BOTH the learned condition vector and the physics prior.
    """
    seq_in = layers.Input(shape=(seq_len, n_rx, n_tx, 2), name="pilot_ls_seq")
    speed_in = layers.Input(shape=(1,), name="speed_norm")
    snr_in = layers.Input(shape=(1,), name="snr_norm")
    delay_in = layers.Input(shape=(1,), name="delay_norm")
    pilot_in = layers.Input(shape=(1,), name="pilot_norm")

    if use_spatial_frontend:
        frames = layers.TimeDistributed(spatial_frontend, name="shared_spatial_frontend")(seq_in)
    else:
        frames = seq_in

    prev_frame = layers.Lambda(lambda t: t[:, -2], name="previous_spatial_frame")(frames)
    last_frame = layers.Lambda(lambda t: t[:, -1], name="last_spatial_frame")(frames)

    # One set of masks is shared by the learned branch and the physics branch.
    speed_used = _masked_scalar_input(speed_in, use_speed, "speed_pg")
    snr_used = _masked_scalar_input(snr_in, use_snr, "snr_pg")
    delay_used = _masked_scalar_input(delay_in, use_delay, "delay_pg")
    pilot_used = _masked_scalar_input(pilot_in, use_pilot, "pilot_pg")

    if use_physics_prior:
        physics_base = PhysicsGuidedJakesLayer(
            carrier_frequency_hz=carrier_frequency_hz,
            history_dt_s=history_dt_s,
            delay_norm_ms=delay_norm_ms,
            name="jakes_correlation_guided_physics_prior",
        )([prev_frame, last_frame, speed_used, delay_used])
    else:
        physics_base = layers.Lambda(lambda t: t, name="physics_prior_disabled_identity")(last_frame)

    x = layers.TimeDistributed(layers.Flatten())(frames)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    cond = layers.Concatenate(name="pg_condition_vector") (
        [speed_used, snr_used, delay_used, pilot_used]
    )
    cond = layers.Dense(64, activation="relu", name="pg_condition_embedding")(cond)
    cond_seq = layers.RepeatVector(seq_len)(cond)
    x = layers.Concatenate(axis=-1)([x, cond_seq])

    for layer_idx in range(max(1, recurrent_layers)):
        return_seq = layer_idx < max(1, recurrent_layers) - 1
        base = layers.LSTM(128, return_sequences=return_seq, name=f"pg_lstm_{layer_idx}")
        x = layers.Bidirectional(base, name=f"pg_bilstm_{layer_idx}")(x) if bidirectional else base(x)

    x = layers.Concatenate(name="pg_fusion_concat")([x, cond, delay_used, snr_used])
    x = layers.Dense(256, activation="relu", name="pg_fusion_dense")(x)
    feat_dim = n_rx * n_tx * 2
    residual_flat = layers.Dense(feat_dim, kernel_initializer="zeros", bias_initializer="zeros", name="pg_residual_delta")(x)
    if use_gate:
        gate = layers.Dense(
            feat_dim,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-1.0),
            name="pg_residual_gate",
        )(x)
        residual_flat = layers.Multiply(name="pg_gated_delta")([residual_flat, gate])
    residual = layers.Reshape((n_rx, n_tx, 2), name="pg_residual_reshape")(residual_flat)
    mu = layers.Add(name="physics_plus_learned_residual")([physics_base, residual])

    if predict_uncertainty:
        logvar_flat = layers.Dense(
            feat_dim,
            bias_initializer=tf.keras.initializers.Constant(initial_log_variance),
            name="pg_log_sigma2",
        )(x)
        logvar = layers.Reshape((n_rx, n_tx, 2), name="pg_log_sigma2_reshape")(logvar_flat)
    else:
        logvar = layers.Lambda(lambda t: tf.zeros_like(t), name="pg_zero_logvar")(mu)

    packed = layers.Concatenate(axis=-1, name="mu_logsigma2_packed")([mu, logvar])
    return models.Model(
        [seq_in, speed_in, snr_in, delay_in, pilot_in],
        packed,
        name="PhysicsGuidedUncertaintyPredictorV6",
    )



def unpack_probabilistic_prediction(pred_packed: Array) -> Tuple[Array, Array]:
    pred = np.asarray(pred_packed)
    c = pred.shape[-1] // 2
    mu = pred[..., :c].astype(np.float32)
    log_sigma2 = np.clip(pred[..., c:], -4.0, 8.0)
    sigma2 = np.exp(log_sigma2).astype(np.float32)
    return mu, sigma2


def predict_physics_guided(model: tf.keras.Model, model_inputs: List[Array]) -> Tuple[Array, Array]:
    return unpack_probabilistic_prediction(model.predict(model_inputs, verbose=0))


def build_multiscale_cnn_baseline(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """GAN-inspired multi-scale CNN baseline (non-adversarial reproduction).

    This intentionally does not claim an exact reproduction of the 2026
    RIS-HSR multi-scale GAN paper. Parallel 3x3/5x5/7x7 branches reproduce the
    multi-receptive-field idea while retaining a stable supervised objective.
    """
    inp = layers.Input(shape=input_shape, name="ls_channel")
    branches = []
    for k in (3, 5, 7):
        b = layers.Conv2D(48, k, padding="same", activation="relu", name=f"ms_k{k}_1")(inp)
        b = layers.Conv2D(48, 3, padding="same", activation="relu", name=f"ms_k{k}_2")(b)
        branches.append(b)
    x = layers.Concatenate(name="ms_concat")(branches)
    x = layers.Conv2D(96, 1, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    noise = layers.Conv2D(input_shape[-1], 3, padding="same", name="ms_noise")(x)
    out = layers.Subtract(name="ms_residual_cleaning")([inp, noise])
    return models.Model(inp, out, name="MultiScaleCNN_GANInspired")


@tf.keras.utils.register_keras_serializable(package="rohitash_custom_layers")
class ContinuousTimeDecay(layers.Layer):
    """Trainable exponential decay used in the GRU-ODE-inspired baseline."""
    def build(self, input_shape):
        self.log_rate = self.add_weight(name="log_decay_rate", shape=(), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # Older snapshots receive stronger decay; spacing is absorbed into the
        # learned positive rate. This is an ODE-RNN approximation, not an exact
        # neural-ODE solver/reproduction.
        n = tf.shape(x)[1]
        ages = tf.cast(tf.range(n - 1, -1, -1), x.dtype)
        rate = tf.nn.softplus(self.log_rate)
        decay = tf.exp(-rate * ages)[None, :, None]
        return x * decay


def build_ode_rnn_baseline(seq_len: int, n_rx: int, n_tx: int) -> tf.keras.Model:
    """GRU-ODE-inspired temporal baseline with continuous-time decay."""
    seq_in = layers.Input(shape=(seq_len, n_rx, n_tx, 2), name="pilot_ls_seq")
    speed_in = layers.Input(shape=(1,), name="speed_norm")
    snr_in = layers.Input(shape=(1,), name="snr_norm")
    delay_in = layers.Input(shape=(1,), name="delay_norm")
    pilot_in = layers.Input(shape=(1,), name="pilot_norm")

    x = layers.TimeDistributed(layers.Flatten())(seq_in)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    x = ContinuousTimeDecay(name="continuous_time_decay")(x)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(128, return_sequences=False)(x)
    cond = layers.Concatenate()([speed_in, snr_in, delay_in, pilot_in])
    cond = layers.Dense(32, activation="relu")(cond)
    x = layers.Concatenate()([x, cond])
    x = layers.Dense(128, activation="relu")(x)
    delta = layers.Dense(n_rx * n_tx * 2, kernel_initializer="zeros", bias_initializer="zeros")(x)
    delta = layers.Reshape((n_rx, n_tx, 2))(delta)
    last = layers.Lambda(lambda t: t[:, -1], name="ode_last_ls")(seq_in)
    out = layers.Add(name="ode_future_residual")([last, delta])
    return models.Model([seq_in, speed_in, snr_in, delay_in, pilot_in], out, name="GRU_ODE_Inspired")


def _temporal_inputs_from_ls_history(
    ls_history_norm: Array,
    speed_kmh: float,
    snr_db: float,
    delay_ms: float,
    pilot_length: int,
    run: RunConfig,
) -> List[Array]:
    seq = np.asarray(ls_history_norm, dtype=np.float32)[None, ...]
    n = 1
    cond = _condition_arrays(
        n,
        np.array([speed_kmh], dtype=np.float32),
        np.array([snr_db], dtype=np.float32),
        np.array([delay_ms], dtype=np.float32),
        np.array([pilot_length], dtype=np.float32),
        run,
    )
    return [seq] + cond


def decide_pilot_length(
    sigma2_prev: float,
    speed_mps: float,
    f_doppler_hz: float,
    delta_t_s: float,
    snr_db: float,
    available_pilot_lengths: Sequence[int],
) -> int:
    """Uncertainty/mobility-aware discrete pilot-length policy.

    The rule is deterministic and transparent. It maps uncertainty, normalized
    Doppler, prediction horizon, and low-SNR risk into a score, then chooses a
    pilot length. It does not use future ground truth.
    """
    choices = sorted(int(v) for v in available_pilot_lengths)
    if not choices:
        raise ValueError("available_pilot_lengths cannot be empty")
    logu = np.log10(max(float(sigma2_prev), 1e-12))
    uncertainty_risk = float(np.clip((logu + 5.0) / 5.0, 0.0, 1.0))
    mobility_risk = float(np.clip(abs(f_doppler_hz) / 1800.0, 0.0, 1.0))
    horizon_risk = float(np.clip(delta_t_s / 0.003, 0.0, 1.0))
    snr_risk = float(np.clip((15.0 - snr_db) / 25.0, 0.0, 1.0))
    speed_risk = float(np.clip(abs(speed_mps) / (500.0 / 3.6), 0.0, 1.0))
    score = 0.42 * uncertainty_risk + 0.18 * mobility_risk + 0.14 * horizon_risk + 0.16 * snr_risk + 0.10 * speed_risk
    idx = int(np.clip(np.floor(score * len(choices)), 0, len(choices) - 1))
    return choices[idx]


def effective_channel_from_components(h_d: Array, h_r: Array, g: Array, phi: Array) -> Array:
    return h_d + h_r @ (phi[:, None] * g)


def generate_effective_channel_components_trajectory(
    cfg: SysConfig,
    rng: np.random.Generator,
    speed_mps: float,
    times_s: Sequence[float],
    *,
    k_offset_db: float = 0.0,
    start_x_m: Optional[float] = None,
) -> Dict[str, Array]:
    """Component-resolved companion generator for IRS-control experiments.

    It follows the same Jakes-correlation-guided temporal evolution used by the
    effective-channel generator but returns H_d(t), H_r(t), and quasi-static G
    so candidate IRS phase vectors can be evaluated without fabricating an
    inverse mapping from H_eff to constituent links.
    """
    times = np.asarray(times_s, dtype=float)
    x0 = float(start_x_m) if start_x_m is not None else float(rng.uniform(cfg.train_x_min_m, cfg.train_x_max_m))

    d_bi = distance(cfg.bs_position, cfg.irs_position)
    g, _ = rician_channel_snapshot(
        cfg.num_irs_elements, cfg.num_bs_antennas, d_bi,
        direction_angle(cfg.bs_position, cfg.irs_position),
        direction_angle(cfg.irs_position, cfg.bs_position),
        _perturbed_k(cfg.rician_k_bs_irs_db, k_offset_db), cfg.pathloss_exp_bs_irs, cfg, rng,
    )
    phi_random = np.exp(1j * rng.uniform(0, 2*np.pi, cfg.num_irs_elements))
    hd = np.zeros((len(times), cfg.num_mcr_antennas, cfg.num_bs_antennas), complex)
    hr = np.zeros((len(times), cfg.num_mcr_antennas, cfg.num_irs_elements), complex)
    nlos_d = nlos_r = None
    phase_d = phase_r = 0.0
    prev = float(times[0])
    for i, t in enumerate(times):
        dt = 0.0 if i == 0 else float(t - prev)
        pos = (x0 + speed_mps * float(t), 0.0)
        ad = direction_angle(cfg.bs_position, pos)
        fd_d = link_doppler_hz(speed_mps, cfg.carrier_frequency_hz, ad)
        if i > 0: phase_d += 2*np.pi*fd_d*dt
        hd_i, nlos_d = rician_channel_snapshot(
            cfg.num_mcr_antennas, cfg.num_bs_antennas, distance(cfg.bs_position, pos), ad,
            direction_angle(pos, cfg.bs_position), _perturbed_k(cfg.rician_k_direct_db, k_offset_db),
            cfg.pathloss_exp_direct, cfg, rng, nlos_d,
            temporal_correlation_jakes(fd_d, dt) if i > 0 else 0.0, phase_d,
        )
        ar = direction_angle(cfg.irs_position, pos)
        fd_r = link_doppler_hz(speed_mps, cfg.carrier_frequency_hz, ar)
        if i > 0: phase_r += 2*np.pi*fd_r*dt
        hr_i, nlos_r = rician_channel_snapshot(
            cfg.num_mcr_antennas, cfg.num_irs_elements, distance(cfg.irs_position, pos), ar,
            direction_angle(pos, cfg.irs_position), _perturbed_k(cfg.rician_k_irs_mcr_db, k_offset_db),
            cfg.pathloss_exp_irs_mcr, cfg, rng, nlos_r,
            temporal_correlation_jakes(fd_r, dt) if i > 0 else 0.0, phase_r,
        )
        if cfg.direct_blockage_db > 0:
            hd_i *= np.sqrt(db_to_linear(-cfg.direct_blockage_db))
        hd[i], hr[i] = hd_i, hr_i
        prev = float(t)
    return {"h_d": hd, "h_r": hr, "g": g, "phi_random": phi_random, "start_x_m": np.array([x0])}


def optimize_irs_phase(
    h_target: Array,
    h_r: Optional[Array] = None,
    g: Optional[Array] = None,
    h_direct: Optional[Array] = None,
    iterations: int = 4,
) -> Array:
    """Dominant-mode phase alignment for a target effective-channel direction.

    h_target selects the desired dominant transmit/receive mode. Current
    constituent channels H_r/G/H_d are required for physically meaningful IRS
    phase design. The update is unit-modulus and deterministic.
    """
    if h_r is None or g is None:
        raise ValueError("h_r and g are required for physical IRS phase optimization")
    if h_direct is None:
        h_direct = np.zeros((h_r.shape[0], g.shape[1]), dtype=np.complex128)
    phi = np.ones(h_r.shape[1], dtype=np.complex128)
    target = np.asarray(h_target, dtype=np.complex128)
    # Keep the dominant mode of the requested target fixed. This is crucial for
    # the predicted-CSI regime: the controller should align the CURRENT IRS
    # element contributions toward the PREDICTED future spatial mode rather
    # than drifting back to a stale/current-channel mode after one iteration.
    u, _, vh = np.linalg.svd(target, full_matrices=False)
    uu, vv = u[:, 0], np.conjugate(vh[0, :])
    direct_scalar = np.vdot(uu, h_direct @ vv)
    contrib = (np.conjugate(uu) @ h_r) * (g @ vv)
    for _ in range(max(1, iterations)):
        if abs(direct_scalar) > 1e-12:
            ref = np.angle(direct_scalar)
        else:
            agg = np.sum(phi * contrib)
            ref = np.angle(agg) if abs(agg) > 1e-12 else 0.0
        phi = np.exp(1j * (ref - np.angle(contrib + 1e-18)))
    return phi.astype(np.complex128)

# =============================================================================
# Empirical LMMSE / Rank-1 SVD baseline
# =============================================================================

@dataclass
class EmpiricalLinearEstimator:
    mean_x: Array
    mean_y: Array
    w: Array
    output_shape: Tuple[int, ...]

    def predict(self, x: Array) -> Array:
        xf = x.reshape(x.shape[0], -1)
        pred = self.mean_y + (xf - self.mean_x) @ self.w.T
        return pred.reshape((x.shape[0],) + self.output_shape)


def fit_empirical_linear_estimator(x_obs: Array, y_true: Array, ridge: float = 1e-7) -> EmpiricalLinearEstimator:
    xf = x_obs.reshape(x_obs.shape[0], -1).astype(np.float64)
    yf = y_true.reshape(y_true.shape[0], -1).astype(np.float64)
    mx, my = xf.mean(axis=0), yf.mean(axis=0)
    xc, yc = xf - mx, yf - my
    cxy = (yc.T @ xc) / max(1, len(xf))
    cxx = (xc.T @ xc) / max(1, len(xf))
    cxx = cxx + ridge * np.eye(cxx.shape[0])
    w = cxy @ np.linalg.pinv(cxx)
    return EmpiricalLinearEstimator(mx, my, w, tuple(y_true.shape[1:]))


def rank1_svd_estimator(x_batch: Array) -> Array:
    out = np.zeros_like(x_batch)
    for i in range(x_batch.shape[0]):
        h = image_to_complex(x_batch[i])
        u, s, vh = np.linalg.svd(h, full_matrices=False)
        h1 = s[0] * np.outer(u[:, 0], vh[0, :])
        out[i] = complex_to_image(h1)
    return out


# =============================================================================
# Training helpers
# =============================================================================


def _callbacks() -> List[tf.keras.callbacks.Callback]:
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=0),
    ]


def train_spatial_network(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    seed: int,
    builder,
    *,
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    epochs: Optional[int] = None,
    verbose: int = 0,
) -> tf.keras.Model:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    ntr = num_train or run.num_train_spatial
    nva = num_val or run.num_val_spatial
    e = epochs or run.epochs_spatial

    xtr, ytr, wtr, _ = build_spatial_dataset(cfg, run, ntr, rng, scale)
    xva, yva, wva, _ = build_spatial_dataset(cfg, run, nva, rng, scale)

    model = builder(tuple(xtr.shape[1:]))
    model.compile(optimizer=tf.keras.optimizers.Adam(run.learning_rate_spatial), loss=per_sample_mse)
    model.fit(
        xtr,
        ytr,
        sample_weight=wtr,
        validation_data=(xva, yva, wva),
        epochs=e,
        batch_size=run.batch_size_spatial,
        callbacks=_callbacks(),
        verbose=verbose,
    )
    return model


def train_cnn_bilstm_baseline(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    seed: int,
    *,
    verbose: int = 0,
) -> tf.keras.Model:
    set_global_seed(seed + 1000)
    rng = np.random.default_rng(seed + 1000)
    xtr, ytr, wtr, _ = build_temporal_dataset(cfg, run, run.num_train_temporal, rng, scale)
    xva, yva, wva, _ = build_temporal_dataset(cfg, run, run.num_val_temporal, rng, scale)

    model = build_cnn_bilstm_baseline(run.seq_len, cfg.num_mcr_antennas, cfg.num_bs_antennas)
    model.compile(optimizer=tf.keras.optimizers.Adam(run.learning_rate_temporal, clipnorm=1.0), loss=per_sample_mse)
    model.fit(
        xtr,
        ytr,
        sample_weight=wtr,
        validation_data=(xva, yva, wva),
        epochs=run.epochs_temporal,
        batch_size=run.batch_size_temporal,
        callbacks=_callbacks(),
        verbose=verbose,
    )
    return model


def train_proposed_temporal(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    seed: int,
    pretrained_spatial: tf.keras.Model,
    *,
    variant: Optional[Dict[str, object]] = None,
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    epochs: Optional[int] = None,
    verbose: int = 0,
) -> tf.keras.Model:
    set_global_seed(seed + 2000)
    rng = np.random.default_rng(seed + 2000)
    ntr = num_train or run.num_train_temporal
    nva = num_val or run.num_val_temporal
    e = epochs or run.epochs_temporal

    xtr, ytr, wtr, _ = build_temporal_dataset(cfg, run, ntr, rng, scale)
    xva, yva, wva, _ = build_temporal_dataset(cfg, run, nva, rng, scale)

    # Clone the spatial network so temporal fine tuning cannot overwrite the
    # separately reported PropSpatial model.
    spatial = tf.keras.models.clone_model(pretrained_spatial)
    spatial.set_weights(pretrained_spatial.get_weights())
    spatial.trainable = False

    kwargs = dict(
        use_speed=True,
        use_snr=True,
        use_delay=True,
        use_pilot=True,
        use_spatial_frontend=True,
        use_gate=True,
        bidirectional=True,
        recurrent_layers=2,
    )
    if variant:
        kwargs.update(variant)

    model = build_proposed_temporal(
        spatial,
        run.seq_len,
        cfg.num_mcr_antennas,
        cfg.num_bs_antennas,
        **kwargs,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(run.learning_rate_temporal, clipnorm=1.0), loss=per_sample_mse)
    model.fit(
        xtr,
        ytr,
        sample_weight=wtr,
        validation_data=(xva, yva, wva),
        epochs=e,
        batch_size=run.batch_size_temporal,
        callbacks=_callbacks(),
        verbose=verbose,
    )

    # Optional low-LR end-to-end fine tuning of the spatial front end.
    if run.temporal_finetune_epochs > 0 and kwargs.get("use_spatial_frontend", True):
        spatial.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(run.learning_rate_temporal * 0.1),
            loss=per_sample_mse,
        )
        model.fit(
            xtr,
            ytr,
            sample_weight=wtr,
            validation_data=(xva, yva, wva),
            epochs=run.temporal_finetune_epochs,
            batch_size=run.batch_size_temporal,
            callbacks=_callbacks(),
            verbose=verbose,
        )
    return model




def train_proposed_temporal_physics_guided(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    seed: int,
    pretrained_spatial: tf.keras.Model,
    *,
    variant: Optional[Dict[str, object]] = None,
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    epochs: Optional[int] = None,
    verbose: int = 0,
) -> tf.keras.Model:
    """Train the default physics-guided uncertainty-aware proposed predictor."""
    set_global_seed(seed + 2500)
    rng = np.random.default_rng(seed + 2500)
    ntr = num_train or run.num_train_temporal
    nva = num_val or run.num_val_temporal
    e = epochs or run.epochs_temporal
    xtr, ytr, wtr, _ = build_temporal_dataset(cfg, run, ntr, rng, scale)
    xva, yva, wva, _ = build_temporal_dataset(cfg, run, nva, rng, scale)

    spatial = tf.keras.models.clone_model(pretrained_spatial)
    spatial.set_weights(pretrained_spatial.get_weights())
    spatial.trainable = False
    kwargs = dict(
        use_speed=True,
        use_snr=True,
        use_delay=True,
        use_pilot=True,
        use_spatial_frontend=True,
        use_gate=True,
        use_physics_prior=True,
        bidirectional=True,
        recurrent_layers=2,
        predict_uncertainty=True,
    )
    if variant:
        kwargs.update(variant)
    model = build_proposed_temporal_physics_guided(
        spatial,
        run.seq_len,
        cfg.num_mcr_antennas,
        cfg.num_bs_antennas,
        carrier_frequency_hz=cfg.carrier_frequency_hz,
        delay_norm_ms=run.delay_train_max_ms,
        history_dt_s=run.history_dt_s,
        initial_log_variance=run.initial_log_variance,
        **kwargs,
    )
    loss = heteroscedastic_nll_loss if run.use_uncertainty_nll else (
        lambda yt, yp: per_sample_mse(yt, yp[..., :2])
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(run.learning_rate_probabilistic, clipnorm=1.0), loss=loss)
    model.fit(
        xtr, ytr, sample_weight=wtr,
        validation_data=(xva, yva, wva),
        epochs=e, batch_size=run.batch_size_temporal,
        callbacks=_callbacks(), verbose=verbose,
    )
    if run.temporal_finetune_epochs > 0 and kwargs.get("use_spatial_frontend", True):
        spatial.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(run.learning_rate_probabilistic * 0.1, clipnorm=1.0),
            loss=loss,
        )
        model.fit(
            xtr, ytr, sample_weight=wtr,
            validation_data=(xva, yva, wva),
            epochs=run.temporal_finetune_epochs,
            batch_size=run.batch_size_temporal,
            callbacks=_callbacks(), verbose=verbose,
        )
    return model


def train_multiscale_cnn_baseline(
    cfg: SysConfig, run: RunConfig, scale: float, seed: int, *, verbose: int = 0
) -> tf.keras.Model:
    return train_spatial_network(
        cfg, run, scale, seed + 3500, build_multiscale_cnn_baseline, verbose=verbose
    )


def train_ode_rnn_baseline(
    cfg: SysConfig, run: RunConfig, scale: float, seed: int, *, verbose: int = 0
) -> tf.keras.Model:
    set_global_seed(seed + 4500)
    rng = np.random.default_rng(seed + 4500)
    xtr, ytr, wtr, _ = build_temporal_dataset(cfg, run, run.num_train_temporal, rng, scale)
    xva, yva, wva, _ = build_temporal_dataset(cfg, run, run.num_val_temporal, rng, scale)
    model = build_ode_rnn_baseline(run.seq_len, cfg.num_mcr_antennas, cfg.num_bs_antennas)
    model.compile(optimizer=tf.keras.optimizers.Adam(run.learning_rate_temporal, clipnorm=1.0), loss=per_sample_mse)
    model.fit(
        xtr, ytr, sample_weight=wtr,
        validation_data=(xva, yva, wva),
        epochs=run.epochs_temporal,
        batch_size=run.batch_size_temporal,
        callbacks=_callbacks(), verbose=verbose,
    )
    return model

# =============================================================================
# Beamforming, BER, and rate with estimated CSI applied to TRUE channel
# =============================================================================


def dominant_beams(h_est: Array) -> Tuple[Array, Array, complex]:
    u, s, vh = np.linalg.svd(h_est, full_matrices=False)
    w = u[:, 0]
    f = np.conjugate(vh[0, :])
    g_est = complex(s[0])
    return w, f, g_est


def effective_gain_on_true_channel(h_true: Array, h_est: Array) -> Tuple[complex, complex]:
    w, f, g_est = dominant_beams(h_est)
    g_actual = np.vdot(w, h_true @ f)
    return complex(g_actual), g_est


def achievable_rate_from_estimate(h_true_norm: Array, h_est_norm: Array, data_snr_db: float) -> float:
    g_actual, _ = effective_gain_on_true_channel(h_true_norm, h_est_norm)
    return float(np.log2(1.0 + db_to_linear(data_snr_db) * (abs(g_actual) ** 2)))


def achievable_rate_perfect(h_true_norm: Array, data_snr_db: float) -> float:
    s = np.linalg.svd(h_true_norm, compute_uv=False)
    return float(np.log2(1.0 + db_to_linear(data_snr_db) * (s[0] ** 2)))


def receiver_noise_power_watt(cfg: SysConfig) -> float:
    noise_dbm = (
        cfg.thermal_noise_density_dbm_hz
        + 10.0 * np.log10(cfg.bandwidth_hz)
        + cfg.noise_figure_db
    )
    return dbm_to_watt(float(noise_dbm))


def achievable_rate_physical_from_estimate(
    h_true_raw: Array, h_est_raw: Array, cfg: SysConfig
) -> float:
    """Single-stream achieved rate using physical raw channel scale."""
    g_actual, _ = effective_gain_on_true_channel(h_true_raw, h_est_raw)
    snr_eff = dbm_to_watt(cfg.transmit_power_dbm) * (abs(g_actual) ** 2) / receiver_noise_power_watt(cfg)
    return float(np.log2(1.0 + snr_eff))


def qpsk_ber_one_channel(
    h_true_norm: Array,
    h_est_norm: Array,
    data_snr_db: float,
    n_symbols: int,
    rng: np.random.Generator,
    *,
    perfect: bool = False,
) -> Tuple[int, int]:
    if perfect:
        u, s, vh = np.linalg.svd(h_true_norm, full_matrices=False)
        w = u[:, 0]
        f = np.conjugate(vh[0, :])
        g_actual = complex(s[0])
        g_equalizer = complex(s[0])
    else:
        w, f, g_equalizer = dominant_beams(h_est_norm)
        g_actual = complex(np.vdot(w, h_true_norm @ f))

    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = ((2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2.0)

    # Receive-vector noise then combining. Since ||w||=1, post-combining noise
    # has the same variance. SNR is defined before channel gain.
    noise_var = 1.0 / db_to_linear(data_snr_db)
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
    )
    rx = g_actual * symbols + noise

    if abs(g_equalizer) < 1e-12:
        return 2 * n_symbols, 2 * n_symbols
    eq = rx / g_equalizer
    bhat = np.stack([(eq.real > 0).astype(int), (eq.imag > 0).astype(int)], axis=1)
    return int(np.sum(bhat != bits)), int(bits.size)


def coherence_symbols(cfg: SysConfig, speed_kmh: float) -> int:
    fd = maximum_doppler_hz(speed_kmh / 3.6, cfg.carrier_frequency_hz)
    if fd <= 0:
        return 10**9
    tc = 0.423 / fd
    return max(1, int(np.floor(tc / cfg.ofdm_symbol_duration_s)))


# =============================================================================
# Complexity measurement
# =============================================================================


def _slice_model_input(inp: ModelInput, n: int) -> ModelInput:
    if isinstance(inp, (list, tuple)):
        return [x[:n] for x in inp]
    return inp[:n]


def estimate_flops(model: tf.keras.Model, sample_input: ModelInput) -> Optional[int]:
    """Best-effort TensorFlow FLOP count for batch size 1."""
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        if isinstance(sample_input, (list, tuple)):
            specs = [
                tf.TensorSpec([1] + list(np.asarray(x).shape[1:]), tf.as_dtype(np.asarray(x).dtype))
                for x in sample_input
            ]

            @tf.function
            def f(*args):
                return model(list(args), training=False)

            concrete = f.get_concrete_function(*specs)
        else:
            arr = np.asarray(sample_input)
            spec = tf.TensorSpec([1] + list(arr.shape[1:]), tf.as_dtype(arr.dtype))

            @tf.function
            def f(x):
                return model(x, training=False)

            concrete = f.get_concrete_function(spec)

        frozen = convert_variables_to_constants_v2(concrete)
        graph_def = frozen.graph.as_graph_def()
        with tf.Graph().as_default() as graph:
            tf.compat.v1.graph_util.import_graph_def(graph_def, name="")
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            prof = tf.compat.v1.profiler.profile(graph=graph, options=opts)
        return int(prof.total_float_ops) if prof is not None else None
    except Exception:
        return None


def measure_complexity(model: tf.keras.Model, sample_input: ModelInput, batch_size: int = 8) -> Dict[str, float]:
    n_params = int(model.count_params())
    batch = _slice_model_input(sample_input, batch_size)
    _ = model.predict(batch, verbose=0)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model.predict(batch, verbose=0)
    elapsed = time.perf_counter() - t0
    latency_ms = 1000.0 * elapsed / (reps * batch_size)
    flops = estimate_flops(model, sample_input)
    return {
        "parameters": n_params,
        "parameter_memory_mb_fp32": float(n_params * 4 / (1024**2)),
        "flops_batch1": float(flops) if flops is not None else float("nan"),
        "inference_ms_per_sample": float(latency_ms),
    }


# =============================================================================
# Evaluation utilities
# =============================================================================


def _predict_temporal(model: tf.keras.Model, model_inputs: List[Array]) -> Array:
    return model.predict(model_inputs, verbose=0)


def _fit_spatial_lmmse_for_condition(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    rng: np.random.Generator,
    snr_db: float,
    speed_kmh: float,
    pilot_length: int,
    *,
    k_offset_db: float = 0.0,
) -> EmpiricalLinearEstimator:
    x, y, _, _ = build_spatial_dataset(
        cfg,
        run,
        run.covariance_samples,
        rng,
        scale,
        fixed_snr_db=snr_db,
        fixed_speed_kmh=speed_kmh,
        fixed_pilot_length=pilot_length,
        k_offset_db=k_offset_db,
    )
    return fit_empirical_linear_estimator(x, y)


def _fit_temporal_lmmse_for_condition(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    rng: np.random.Generator,
    snr_db: float,
    speed_kmh: float,
    delay_ms: float,
    pilot_length: int,
    *,
    k_offset_db: float = 0.0,
) -> EmpiricalLinearEstimator:
    x, y, _, _ = build_temporal_dataset(
        cfg,
        run,
        run.covariance_samples,
        rng,
        scale,
        fixed_snr_db=snr_db,
        fixed_speed_kmh=speed_kmh,
        fixed_delay_ms=delay_ms,
        fixed_pilot_length=pilot_length,
        k_offset_db=k_offset_db,
    )
    return fit_empirical_linear_estimator(x[0], y)


# =============================================================================
# Main multi-seed NMSE vs SNR
# =============================================================================


# =============================================================================
# Speed / unseen-speed generalization
# =============================================================================


# =============================================================================
# Same-trajectory channel aging
# =============================================================================


# =============================================================================
# Pilot efficiency and net spectral efficiency
# =============================================================================


# =============================================================================
# BER and spectral efficiency vs SNR
# =============================================================================


# =============================================================================
# K-factor mismatch, phase errors, and unseen-speed robustness
# =============================================================================


# =============================================================================
# IRS-size scalability with physical-power reporting
# =============================================================================


# =============================================================================
# Temporal novelty ablation
# =============================================================================


# =============================================================================
# V6 evaluations: stronger baselines + closed-loop innovations
# =============================================================================


def evaluate_nmse_vs_snr(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    models_by_seed: List[Dict[str, object]],
    *,
    temporal_delay_ms: float = 0.0,
    include_spatial: bool = True,
) -> Dict[str, Dict[str, List[float]]]:
    """NMSE vs SNR for the full baseline roster.

    V6: ``temporal_delay_ms`` controls the prediction horizon used for every
    TEMPORAL method (Temporal LMMSE, CNN-BiLSTM-inspired, GRU-ODE-inspired,
    Proposed). The default (0.0) reproduces the original same-instant
    comparison; calling this with ``temporal_delay_ms=1.0`` (or any nonzero
    value) instead reports NMSE vs SNR at a genuine prediction horizon, which
    is the operating point that actually exercises the proposed model's
    predictive claim rather than its identity/near-identity behavior at
    delay=0. ``include_spatial=False`` skips the (delay-independent) spatial
    baselines so the nonzero-delay call only repeats the temporal roster.
    """
    spatial_methods = [
        "LS",
        "Rank-1 SVD baseline",
        "Matched LMMSE",
        "Mismatched LMMSE",
        "CDRN-inspired baseline",
        "Prop-Spatial",
    ]
    if run.include_gan_baseline:
        spatial_methods.insert(5, "MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)")
    temporal_methods = [
        "Temporal LMMSE",
        "CNN-BiLSTM-inspired baseline",
        "Proposed physics-guided temporal",
    ]
    if run.include_ode_baseline:
        temporal_methods.insert(-1, "GRU-ODE-inspired temporal baseline")
    methods = (spatial_methods if include_spatial else []) + temporal_methods
    raw = {m: {float(s): [] for s in run.snr_points_db} for m in methods}

    for bundle in models_by_seed:
        seed = int(bundle["seed"])
        for snr in run.snr_points_db:
            rng = np.random.default_rng(
                seed + 10_000 + int((snr + 20) * 10) + int(temporal_delay_ms * 1000)
            )
            if include_spatial:
                x_sp, y_sp, _, _ = build_spatial_dataset(
                    cfg, run, run.num_test, rng, scale,
                    fixed_snr_db=snr,
                    fixed_speed_kmh=run.fixed_eval_speed_kmh,
                    fixed_pilot_length=run.main_pilot_length,
                )
                matched = _fit_spatial_lmmse_for_condition(
                    cfg, run, scale, rng, snr, run.fixed_eval_speed_kmh, run.main_pilot_length
                )
                mismatched = _fit_spatial_lmmse_for_condition(
                    cfg, run, scale, rng, snr, run.fixed_eval_speed_kmh,
                    run.main_pilot_length, k_offset_db=run.lmmse_mismatch_k_offset_db,
                )
                raw["LS"][float(snr)].append(nmse_db(y_sp, x_sp))
                raw["Rank-1 SVD baseline"][float(snr)].append(nmse_db(y_sp, rank1_svd_estimator(x_sp)))
                raw["Matched LMMSE"][float(snr)].append(nmse_db(y_sp, matched.predict(x_sp)))
                raw["Mismatched LMMSE"][float(snr)].append(nmse_db(y_sp, mismatched.predict(x_sp)))
                raw["CDRN-inspired baseline"][float(snr)].append(
                    nmse_db(y_sp, bundle["cdrn"].predict(x_sp, verbose=0))
                )
                raw["Prop-Spatial"][float(snr)].append(
                    nmse_db(y_sp, bundle["spatial"].predict(x_sp, verbose=0))
                )
                if run.include_gan_baseline:
                    raw["MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)"][float(snr)].append(
                        nmse_db(y_sp, bundle["multiscale"].predict(x_sp, verbose=0))
                    )

            x_t, y_t, _, _ = build_temporal_dataset(
                cfg, run, min(run.num_test, 800), rng, scale,
                fixed_snr_db=snr,
                fixed_speed_kmh=run.fixed_eval_speed_kmh,
                fixed_delay_ms=temporal_delay_ms,
                fixed_pilot_length=run.main_pilot_length,
            )
            temporal_lmmse = _fit_temporal_lmmse_for_condition(
                cfg, run, scale, rng, snr, run.fixed_eval_speed_kmh, temporal_delay_ms, run.main_pilot_length
            )
            raw["Temporal LMMSE"][float(snr)].append(nmse_db(y_t, temporal_lmmse.predict(x_t[0])))
            raw["CNN-BiLSTM-inspired baseline"][float(snr)].append(
                nmse_db(y_t, _predict_temporal(bundle["cnn_bilstm"], x_t))
            )
            if run.include_ode_baseline:
                raw["GRU-ODE-inspired temporal baseline"][float(snr)].append(
                    nmse_db(y_t, _predict_temporal(bundle["ode_rnn"], x_t))
                )
            raw["Proposed physics-guided temporal"][float(snr)].append(
                nmse_db(y_t, _predict_temporal(bundle["proposed"], x_t))
            )

    out = {}
    for m in methods:
        stats = [summarize_seed_values(raw[m][float(s)]) for s in run.snr_points_db]
        out[m] = {
            "mean": [z["mean"] for z in stats],
            "std": [z["std"] for z in stats],
            "ci95": [z["ci95"] for z in stats],
        }
    return out


def evaluate_nmse_vs_speed(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    models_by_seed: List[Dict[str, object]],
) -> Dict[str, Dict[str, List[float]]]:
    # Prediction NMSE versus speed with condition-matched and frozen LMMSE.
    methods = [
        "Stale Prop-Spatial",
        "Temporal LMMSE (condition-matched)",
        "Frozen Temporal LMMSE (nominal 1 ms)",
        "CNN-BiLSTM-inspired baseline",
        "Proposed physics-guided temporal",
    ]
    if run.include_ode_baseline:
        methods.insert(-1, "GRU-ODE-inspired temporal baseline")
    if run.include_gan_baseline:
        methods.insert(1, "Stale MultiScaleCNN GAN-inspired baseline")

    values = {m: {float(v): [] for v in run.speed_points_kmh} for m in methods}

    for bundle in models_by_seed:
        seed = int(bundle["seed"])

        rng_frozen = np.random.default_rng(seed + 19_500)
        frozen_tl = _fit_temporal_lmmse_for_condition(
            cfg, run, scale, rng_frozen,
            run.fixed_eval_snr_db,
            run.fixed_eval_speed_kmh,
            run.headline_prediction_delay_ms,
            run.main_pilot_length,
        )

        for speed in run.speed_points_kmh:
            rng = np.random.default_rng(seed + 20_000 + int(speed))
            x_t, y_t, _, _ = build_temporal_dataset(
                cfg, run, min(run.num_test, 800), rng, scale,
                fixed_snr_db=run.fixed_eval_snr_db,
                fixed_speed_kmh=speed,
                fixed_delay_ms=run.speed_eval_prediction_delay_ms,
                fixed_pilot_length=run.main_pilot_length,
            )
            cur = x_t[0][:, -1]

            values["Stale Prop-Spatial"][float(speed)].append(
                nmse_db(y_t, bundle["spatial"].predict(cur, verbose=0))
            )
            if run.include_gan_baseline:
                values["Stale MultiScaleCNN GAN-inspired baseline"][float(speed)].append(
                    nmse_db(y_t, bundle["multiscale"].predict(cur, verbose=0))
                )

            matched_tl = _fit_temporal_lmmse_for_condition(
                cfg, run, scale, rng,
                run.fixed_eval_snr_db,
                speed,
                run.speed_eval_prediction_delay_ms,
                run.main_pilot_length,
            )
            values["Temporal LMMSE (condition-matched)"][float(speed)].append(
                nmse_db(y_t, matched_tl.predict(x_t[0]))
            )
            values["Frozen Temporal LMMSE (nominal 1 ms)"][float(speed)].append(
                nmse_db(y_t, frozen_tl.predict(x_t[0]))
            )
            values["CNN-BiLSTM-inspired baseline"][float(speed)].append(
                nmse_db(y_t, _predict_temporal(bundle["cnn_bilstm"], x_t))
            )
            if run.include_ode_baseline:
                values["GRU-ODE-inspired temporal baseline"][float(speed)].append(
                    nmse_db(y_t, _predict_temporal(bundle["ode_rnn"], x_t))
                )
            values["Proposed physics-guided temporal"][float(speed)].append(
                nmse_db(y_t, _predict_temporal(bundle["proposed"], x_t))
            )

    out = {}
    for m in methods:
        stats = [summarize_seed_values(values[m][float(v)]) for v in run.speed_points_kmh]
        out[m] = {
            "mean": [z["mean"] for z in stats],
            "std": [z["std"] for z in stats],
            "ci95": [z["ci95"] for z in stats],
        }
    return out

def evaluate_channel_aging(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    models_by_seed: List[Dict[str, object]],
) -> Dict[str, Dict[str, List[float]]]:
    """CSI-aging comparison at a fixed (speed, SNR) over increasing delay.

    V6 adds two comparators that were missing from the aging figure:
      - "Temporal LMMSE": the same per-condition-refitted empirical temporal
        LMMSE used in the SNR/pilot-efficiency plots, now also evaluated here
        so the strongest classical predictor is visible on the aging curve.
      - "Frozen Temporal LMMSE (nominal 1 ms)": a single temporal LMMSE fit ONCE
        at delay=0 for the reference (speed, SNR) and then reused, unmodified,
        at every other delay. This is the practically deployable variant: a
        real system does not re-fit a fresh LMMSE filter for every exact
        aging delay it might encounter, so this frozen variant is a fairer
        stress test than a filter that is quietly re-optimized per condition.
    """
    methods = [
        "Stale LS",
        "Stale Matched LMMSE",
        "Stale CDRN-inspired baseline",
        "Stale Prop-Spatial",
        "Temporal LMMSE",
        "Frozen Temporal LMMSE (nominal 1 ms)",
        "CNN-BiLSTM-inspired predictor",
        "Proposed physics-guided predictor",
    ]
    if run.include_gan_baseline:
        methods.insert(4, "Stale MultiScaleCNN GAN-inspired baseline")
    if run.include_ode_baseline:
        methods.insert(-1, "GRU-ODE-inspired predictor")
    raw = {m: {float(d): [] for d in run.aging_delays_ms} for m in methods}
    jref = {float(d): [] for d in run.aging_delays_ms}

    for bundle in models_by_seed:
        seed = int(bundle["seed"])
        rng_nominal = np.random.default_rng(seed + 30_500)
        nominal_temporal_lmmse = _fit_temporal_lmmse_for_condition(
            cfg,
            run,
            scale,
            rng_nominal,
            run.fixed_eval_snr_db,
            run.fixed_eval_speed_kmh,
            run.headline_prediction_delay_ms,
            run.main_pilot_length,
        )
        for delay in run.aging_delays_ms:
            rng = np.random.default_rng(seed + 30_000 + int(delay * 1000))
            x_t, y_t, _, _ = build_temporal_dataset(
                cfg,
                run,
                min(run.num_test, 800),
                rng,
                scale,
                fixed_snr_db=run.fixed_eval_snr_db,
                fixed_speed_kmh=run.fixed_eval_speed_kmh,
                fixed_delay_ms=delay,
                fixed_pilot_length=run.main_pilot_length,
                return_raw=True,
            )
            cur = x_t[0][:, -1]
            lmmse = _fit_spatial_lmmse_for_condition(
                cfg,
                run,
                scale,
                rng,
                run.fixed_eval_snr_db,
                run.fixed_eval_speed_kmh,
                run.main_pilot_length,
            )
            raw["Stale LS"][float(delay)].append(nmse_db(y_t, cur))
            raw["Stale Matched LMMSE"][float(delay)].append(
                nmse_db(y_t, lmmse.predict(cur))
            )
            raw["Stale CDRN-inspired baseline"][float(delay)].append(
                nmse_db(y_t, bundle["cdrn"].predict(cur, verbose=0))
            )
            raw["Stale Prop-Spatial"][float(delay)].append(
                nmse_db(y_t, bundle["spatial"].predict(cur, verbose=0))
            )
            if run.include_gan_baseline:
                raw["Stale MultiScaleCNN GAN-inspired baseline"][float(delay)].append(
                    nmse_db(y_t, bundle["multiscale"].predict(cur, verbose=0))
                )
            temporal_lmmse_delay = _fit_temporal_lmmse_for_condition(
                cfg,
                run,
                scale,
                rng,
                run.fixed_eval_snr_db,
                run.fixed_eval_speed_kmh,
                delay,
                run.main_pilot_length,
            )
            raw["Temporal LMMSE"][float(delay)].append(
                nmse_db(y_t, temporal_lmmse_delay.predict(x_t[0]))
            )
            raw["Frozen Temporal LMMSE (nominal 1 ms)"][float(delay)].append(
                nmse_db(y_t, nominal_temporal_lmmse.predict(x_t[0]))
            )
            raw["CNN-BiLSTM-inspired predictor"][float(delay)].append(
                nmse_db(y_t, _predict_temporal(bundle["cnn_bilstm"], x_t))
            )
            if run.include_ode_baseline:
                raw["GRU-ODE-inspired predictor"][float(delay)].append(
                    nmse_db(y_t, _predict_temporal(bundle["ode_rnn"], x_t))
                )
            raw["Proposed physics-guided predictor"][float(delay)].append(
                nmse_db(y_t, _predict_temporal(bundle["proposed"], x_t))
            )
            rho = temporal_correlation_jakes(
                maximum_doppler_hz(
                    run.fixed_eval_speed_kmh / 3.6, cfg.carrier_frequency_hz
                ),
                delay / 1000.0,
            )
            base = (
                nmse_linear(y_t, cur)
                if delay == 0
                else db_to_linear(raw["Stale LS"][0.0][-1])
            )
            jref[float(delay)].append(
                linear_to_db(base + max(0.0, 2.0 - 2.0 * rho))
            )

    out: Dict[str, Dict[str, List[float]]] = {}
    for m in methods:
        stats = [summarize_seed_values(raw[m][float(d)]) for d in run.aging_delays_ms]
        out[m] = {
            "mean": [z["mean"] for z in stats],
            "std": [z["std"] for z in stats],
            "ci95": [z["ci95"] for z in stats],
        }
    stats = [summarize_seed_values(jref[float(d)]) for d in run.aging_delays_ms]
    out["Jakes correlation reference"] = {
        "mean": [z["mean"] for z in stats],
        "std": [z["std"] for z in stats],
        "ci95": [z["ci95"] for z in stats],
    }
    return out



def evaluate_uncertainty_calibration(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    probabilistic_model: tf.keras.Model,
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(31_337)
    x, y, _, _ = build_temporal_dataset(
        cfg, run, min(run.num_test, 1000), rng, scale,
        fixed_snr_db=run.fixed_eval_snr_db,
        fixed_speed_kmh=run.fixed_eval_speed_kmh,
        fixed_delay_ms=1.0,
        fixed_pilot_length=run.main_pilot_length,
    )
    mu, sigma2 = predict_physics_guided(probabilistic_model, x)
    sample_err = np.sqrt(np.mean((y - mu) ** 2, axis=(1, 2, 3)))
    sample_sigma = np.sqrt(np.mean(sigma2, axis=(1, 2, 3)))
    corr = float(np.corrcoef(sample_sigma, sample_err)[0, 1]) if len(sample_err) > 1 else 0.0

    order = np.argsort(sample_sigma)
    bins = np.array_split(order, max(2, run.uncertainty_bins))
    pred_bin, err_bin, weights = [], [], []
    for ids in bins:
        if len(ids) == 0:
            continue
        pred_bin.append(float(np.mean(sample_sigma[ids])))
        err_bin.append(float(np.mean(sample_err[ids])))
        weights.append(len(ids) / len(sample_err))
    ece = float(np.sum(np.asarray(weights) * np.abs(np.asarray(pred_bin) - np.asarray(err_bin))))

    sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    abs_resid = np.abs(y - mu)
    coverage68 = float(np.mean(abs_resid <= sigma))
    coverage95 = float(np.mean(abs_resid <= 1.96 * sigma))

    if output_path is not None:
        plt.figure(figsize=(7.5, 5.5))
        plt.plot(pred_bin, err_bin, "o-", label="Empirical error")
        lim = max(max(pred_bin, default=1.0), max(err_bin, default=1.0))
        plt.plot([0, lim], [0, lim], "k--", label="Ideal calibration")
        plt.xlabel("Predicted RMS uncertainty")
        plt.ylabel("Empirical RMS error")
        plt.title("Uncertainty calibration")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=250)
        plt.close()
    return {
        "uncertainty_error_correlation": corr,
        "ece_rms": ece,
        "coverage_68": coverage68,
        "coverage_95": coverage95,
        "predicted_bin_rms": pred_bin,
        "empirical_bin_rms": err_bin,
    }


def evaluate_adaptive_pilot_policy(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    probabilistic_model: tf.keras.Model,
) -> Dict[str, object]:
    """Evaluate a true streaming uncertainty-driven adaptive pilot policy.

    V6 fixes the former retroactive-history issue: once an LS observation has
    been acquired, its pilot length and value are retained. The controller
    chooses only the pilot length for the NEXT observation. No old channel
    state is re-observed using a newly selected tau.
    """
    choices_source = run.pilot_lengths if run.adaptive_allow_unseen_pilot_lengths else run.training_pilot_lengths
    choices = tuple(sorted(set(int(t) for t in choices_source)))
    if not choices:
        raise ValueError("No pilot lengths are available for the adaptive policy")

    speed = run.fixed_eval_speed_kmh
    speed_mps = speed / 3.6
    snr = run.fixed_eval_snr_db
    dt = run.history_dt_s
    delay_ms = dt * 1000.0
    coh = coherence_symbols(cfg, speed)
    policies = ["adaptive"] + [f"fixed_tau_{t}" for t in choices]
    errors = {k: [] for k in policies}
    acquired_taus = {k: [] for k in policies}
    objectives = {k: [] for k in policies}

    for traj_idx in range(run.adaptive_policy_trajectories):
        rng_h = np.random.default_rng(100_000 + traj_idx)
        times = np.arange(run.adaptive_policy_steps + 1, dtype=float) * dt
        htraj = generate_effective_channel_trajectory(cfg, rng_h, speed_mps, times)

        def run_policy(policy_tau: Optional[int]):
            # Initialization uses the fixed tau for fixed policies. Adaptive
            # operation starts from the configured main pilot length if it is
            # available, otherwise from the nearest allowed trained length.
            if policy_tau is not None:
                init_tau = int(policy_tau)
            else:
                init_tau = min(choices, key=lambda t: abs(t - run.main_pilot_length))

            history: List[Array] = []
            local_acquired: List[int] = []
            for j in range(run.seq_len):
                tau_j = int(policy_tau) if policy_tau is not None else init_tau
                rng_n = np.random.default_rng(200_000 + traj_idx * 100_000 + j * 100 + tau_j)
                _, hls, _ = pilot_observation_and_ls(htraj[j], snr, tau_j, rng_n)
                history.append(complex_to_image(hls / scale))
                local_acquired.append(tau_j)

            current_tau = local_acquired[-1]
            per_err: List[float] = []
            per_obj: List[float] = []

            for k in range(run.seq_len - 1, run.adaptive_policy_steps):
                inputs = _temporal_inputs_from_ls_history(
                    np.asarray(history), speed, snr, delay_ms, current_tau, run
                )
                mu, sig2 = predict_physics_guided(probabilistic_model, inputs)
                target = complex_to_image(htraj[k + 1] / scale)[None, ...]
                e = nmse_linear(target, mu)
                sigma2_now = float(np.mean(sig2))
                fd = maximum_doppler_hz(speed_mps, cfg.carrier_frequency_hz)

                if policy_tau is None:
                    tau_next = decide_pilot_length(
                        sigma2_now, speed_mps, fd, dt, snr, choices
                    )
                else:
                    tau_next = int(policy_tau)

                # Pair the prediction error with the resource decision made
                # for the next update interval. This is the intended
                # error-overhead tradeoff; only actually acquired pilots are
                # included in the average pilot-length headline.
                per_err.append(e)
                per_obj.append(e + run.adaptive_pilot_lambda * (tau_next / max(coh, 1)))

                # Acquire exactly one NEW pilot block for the next state, then
                # roll the temporal history forward. Do not acquire after the
                # terminal prediction because there is no following update.
                if k + 1 < run.adaptive_policy_steps:
                    rng_n = np.random.default_rng(
                        210_000 + traj_idx * 100_000 + (k + 1) * 100 + tau_next
                    )
                    _, hls_next, _ = pilot_observation_and_ls(
                        htraj[k + 1], snr, tau_next, rng_n
                    )
                    history = history[1:] + [complex_to_image(hls_next / scale)]
                    local_acquired.append(int(tau_next))
                    current_tau = int(tau_next)

            return per_err, local_acquired, per_obj

        for name in policies:
            fixed_tau = None if name == "adaptive" else int(name.split("_")[-1])
            e, t, o = run_policy(fixed_tau)
            errors[name].extend(e)
            acquired_taus[name].extend(t)
            objectives[name].extend(o)

    summary: Dict[str, object] = {}
    for name in policies:
        summary[name] = {
            "nmse_db": linear_to_db(float(np.mean(errors[name]))),
            "average_pilot_length": float(np.mean(acquired_taus[name])),
            "objective": float(np.mean(objectives[name])),
            "num_acquired_pilot_blocks": int(len(acquired_taus[name])),
        }

    adaptive_nmse = float(summary["adaptive"]["nmse_db"])
    qualifying = [
        (float(summary[f"fixed_tau_{t}"]["average_pilot_length"]), int(t))
        for t in choices
        if float(summary[f"fixed_tau_{t}"]["nmse_db"]) <= adaptive_nmse + 0.25
    ]
    if qualifying:
        # Shortest fixed policy that genuinely matches/beats adaptive NMSE.
        # Negative savings remain possible and are intentionally reported as a
        # FAIL rather than hidden by a sign manipulation.
        ref_tau, ref_policy_tau = min(qualifying, key=lambda z: z[0])
        adaptive_tau = float(summary["adaptive"]["average_pilot_length"])
        savings = 100.0 * (ref_tau - adaptive_tau) / ref_tau
        ref_name = f"fixed_tau_{ref_policy_tau}"
    else:
        ref_tau, savings, ref_name = float("nan"), float("nan"), None

    best_objective_policy = min(policies, key=lambda p: float(summary[p]["objective"]))
    summary["headline"] = {
        "adaptive_nmse_db": adaptive_nmse,
        "adaptive_average_tau": float(summary["adaptive"]["average_pilot_length"]),
        "matched_nmse_fixed_tau_reference": ref_tau,
        "matched_nmse_fixed_policy": ref_name,
        "pilot_saving_percent_at_matched_nmse": savings,
        "pilot_saving_is_positive": bool(np.isfinite(savings) and savings > 0.0),
        "best_objective_policy": best_objective_policy,
        "adaptive_is_best_objective": bool(best_objective_policy == "adaptive"),
        "policy_pilot_set": list(choices),
        "policy_uses_only_training_pilot_lengths": bool(not run.adaptive_allow_unseen_pilot_lengths),
    }
    return summary



def evaluate_predicted_csi_irs_optimization(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    probabilistic_model: tf.keras.Model,
) -> Dict[str, object]:
    """Compare IRS control policies using estimated/predicted CSI for beams.

    The controller has the current constituent-channel phase sensitivity
    (H_r(t), G), as required to change Phi, but never uses future H_r/H_d when
    choosing Phi. Future performance is evaluated on the true future channel.
    Predicted-CSI control uses the proposed future H_eff prediction to select
    the desired spatial mode and a current-Jacobian approximation to map a
    phase change into an estimated future effective channel.
    """
    regimes = [
        "Random IRS phase",
        "Static optimized IRS",
        "Stale-CSI optimized IRS",
        "Predicted-CSI optimized IRS",
    ]
    rates = {r: [] for r in regimes}
    speed = run.fixed_eval_speed_kmh
    snr = run.fixed_eval_snr_db
    dt = run.history_dt_s
    delay_ms = dt * 1000.0
    tau = run.main_pilot_length

    for idx in range(run.irs_control_trajectories):
        rng = np.random.default_rng(300_000 + idx)
        times = np.arange(run.seq_len + 1, dtype=float) * dt
        comp = generate_effective_channel_components_trajectory(
            cfg, rng, speed / 3.6, times
        )
        hd, hr, g, phi0 = comp["h_d"], comp["h_r"], comp["g"], comp["phi_random"]
        hist_true = [
            effective_channel_from_components(hd[t], hr[t], g, phi0)
            for t in range(run.seq_len)
        ]
        ls_hist = []
        for t, h in enumerate(hist_true):
            _, hls, _ = pilot_observation_and_ls(
                h, snr, tau, np.random.default_rng(310_000 + idx * 100 + t)
            )
            ls_hist.append(complex_to_image(hls / scale))
        inputs = _temporal_inputs_from_ls_history(
            np.asarray(ls_hist), speed, snr, delay_ms, tau, run
        )
        mu, _ = predict_physics_guided(probabilistic_model, inputs)
        h_pred_phi0 = image_to_complex(mu[0]) * scale
        h_current_est_phi0 = image_to_complex(ls_hist[-1]) * scale
        h_first_est_phi0 = image_to_complex(ls_hist[0]) * scale

        # Only current/past constituent sensitivities are available to the
        # controller. hr[-1]/hd[-1] are reserved exclusively for evaluation.
        hr_current, hd_current = hr[-2], hd[-2]
        hr_first, hd_first = hr[0], hd[0]

        phi_static = optimize_irs_phase(
            h_first_est_phi0, hr_first, g, hd_first, run.irs_phase_iterations
        )
        phi_stale = optimize_irs_phase(
            h_current_est_phi0, hr_current, g, hd_current, run.irs_phase_iterations
        )
        phi_pred = optimize_irs_phase(
            h_pred_phi0, hr_current, g, hd_current, run.irs_phase_iterations
        )

        candidates = {
            "Random IRS phase": phi0,
            "Static optimized IRS": phi_static,
            "Stale-CSI optimized IRS": phi_stale,
            "Predicted-CSI optimized IRS": phi_pred,
        }

        # Estimated channel under a candidate Phi using the current phase
        # sensitivity. This avoids perfect-future-CSI beamforming.
        def estimate_under_candidate(base_est: Array, phi_candidate: Array) -> Array:
            delta_phi = phi_candidate - phi0
            return base_est + hr_current @ (delta_phi[:, None] * g)

        estimated_for_beams = {
            "Random IRS phase": h_current_est_phi0,
            "Static optimized IRS": estimate_under_candidate(h_first_est_phi0, phi_static),
            "Stale-CSI optimized IRS": estimate_under_candidate(h_current_est_phi0, phi_stale),
            "Predicted-CSI optimized IRS": estimate_under_candidate(h_pred_phi0, phi_pred),
        }

        for name, phi in candidates.items():
            h_future_true = effective_channel_from_components(hd[-1], hr[-1], g, phi)
            h_est = estimated_for_beams[name]
            rates[name].append(
                achievable_rate_physical_from_estimate(h_future_true, h_est, cfg)
            )

    summary = {k: summarize_seed_values(v) for k, v in rates.items()}
    summary["headline"] = {
        "predicted_minus_stale_rate_bps_hz": float(
            np.mean(rates["Predicted-CSI optimized IRS"])
            - np.mean(rates["Stale-CSI optimized IRS"])
        ),
        "predicted_minus_random_rate_bps_hz": float(
            np.mean(rates["Predicted-CSI optimized IRS"])
            - np.mean(rates["Random IRS phase"])
        ),
        "beamforming_uses_future_ground_truth": False,
        "controller_future_constituent_csi_used": False,
        "controller_model": "predicted H_eff target + current H_r/G phase-sensitivity approximation",
    }
    return summary



def evaluate_closed_loop_chain(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    probabilistic_model: tf.keras.Model,
    *,
    out_dir: Optional[Path] = None,
) -> Dict[str, object]:
    calibration = evaluate_uncertainty_calibration(
        cfg, run, scale, probabilistic_model,
        output_path=(out_dir / "11_uncertainty_calibration.png") if out_dir else None,
    )
    adaptive = evaluate_adaptive_pilot_policy(cfg, run, scale, probabilistic_model)
    irs = evaluate_predicted_csi_irs_optimization(cfg, run, scale, probabilistic_model)
    result = {"uncertainty_calibration": calibration, "adaptive_pilots": adaptive, "predicted_csi_irs": irs}
    if out_dir is not None:
        labels = ["Uncertainty-error corr.", "Pilot saving (%)", "Predicted vs stale IRS rate gain"]
        vals = [
            calibration["uncertainty_error_correlation"],
            adaptive["headline"]["pilot_saving_percent_at_matched_nmse"],
            irs["headline"]["predicted_minus_stale_rate_bps_hz"],
        ]
        plt.figure(figsize=(9, 5.5))
        plt.bar(np.arange(3), vals)
        plt.xticks(np.arange(3), labels, rotation=15, ha="right")
        plt.title("Closed-loop chain summary (computed, not hard-coded)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "12_closed_loop_chain_summary.png", dpi=250)
        plt.close()
    return result


# =============================================================================
# Wideband OFDM extension and cross-scenario generalization
# =============================================================================


def generate_effective_channel_trajectory_ofdm(
    cfg: SysConfig,
    rng: np.random.Generator,
    speed_mps: float,
    times_s: Sequence[float],
    *,
    k_offset_db: float = 0.0,
    start_x_m: Optional[float] = None,
) -> Array:
    """Stylized wideband MIMO-OFDM effective-channel trajectory H[t,k].

    V6 uses ONE physically common IRS phase configuration and one quasi-static
    BS-IRS matrix across every multipath tap. Independent complex tap gains
    create a frequency-selective delay profile without the unphysical behavior
    of regenerating a different IRS control state for each delay tap.

    The model remains a controlled wideband extension, not a standardized
    measured railway channel model.
    """
    K = cfg.num_subcarriers
    L = cfg.num_channel_taps
    if K < 2 or L < 1:
        raise ValueError("num_subcarriers>=2 and num_channel_taps>=1 required")

    times = np.asarray(times_s, dtype=float)
    x0 = float(start_x_m) if start_x_m is not None else float(
        rng.uniform(cfg.train_x_min_m, cfg.train_x_max_m)
    )
    delays = np.linspace(0.0, cfg.max_excess_delay_s, L)
    pdp = np.exp(-np.arange(L, dtype=float))
    pdp /= np.sum(pdp)
    freqs = np.arange(K, dtype=float) * cfg.subcarrier_spacing_hz

    # One common component trajectory -> one common G and one common Phi.
    comp = generate_effective_channel_components_trajectory(
        cfg, rng, speed_mps, times, k_offset_db=k_offset_db, start_x_m=x0
    )
    hd = comp["h_d"]
    hr = comp["h_r"]
    g = comp["g"]
    phi = comp["phi_random"]
    reflected = np.einsum("tnm,mq->tnq", hr, phi[:, None] * g)

    out = np.zeros(
        (len(times), K, cfg.num_mcr_antennas, cfg.num_bs_antennas),
        dtype=np.complex128,
    )

    # First tap retains a strong common component; later taps receive
    # independent complex gains for direct and reflected contributions. This
    # yields nontrivial H[k] while preserving a common IRS control state.
    for ell in range(L):
        if ell == 0:
            alpha_d = 1.0 + 0.0j
            alpha_r = 1.0 + 0.0j
        else:
            alpha_d = complex_gaussian((1,), rng)[0]
            alpha_r = complex_gaussian((1,), rng)[0]
        tap = np.sqrt(pdp[ell]) * (alpha_d * hd + alpha_r * reflected)
        phase_f = np.exp(-1j * 2.0 * np.pi * freqs * delays[ell])
        out += tap[:, None, :, :] * phase_f[None, :, None, None]
    return out



def pilot_observation_and_ls_ofdm(
    h_true_k: Array,
    snr_db: float,
    pilot_length: int,
    pilot_subcarrier_stride: int,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    """Comb-subcarrier pilot observation plus linear frequency interpolation."""
    K = h_true_k.shape[0]
    pilot_idx = np.arange(0, K, max(1, int(pilot_subcarrier_stride)), dtype=int)
    ls_p = np.zeros((len(pilot_idx),) + h_true_k.shape[1:], dtype=np.complex128)
    for ii, k in enumerate(pilot_idx):
        _, ls_p[ii], _ = pilot_observation_and_ls(h_true_k[k], snr_db, pilot_length, rng)
    ls_full = np.zeros_like(h_true_k)
    grid = np.arange(K)
    for r in range(h_true_k.shape[1]):
        for t in range(h_true_k.shape[2]):
            re = np.interp(grid, pilot_idx, ls_p[:, r, t].real)
            im = np.interp(grid, pilot_idx, ls_p[:, r, t].imag)
            ls_full[:, r, t] = re + 1j * im
    return pilot_idx, ls_full


def _replicate_temporal_conditions_for_subcarriers(
    seq_k: Array, speed_kmh: float, snr_db: float, delay_ms: float,
    pilot_length: int, run: RunConfig,
) -> List[Array]:
    # seq_k shape: (K, seq_len, Nr, Nt, 2)
    K = seq_k.shape[0]
    cond = _condition_arrays(
        K,
        np.full(K, speed_kmh, dtype=np.float32),
        np.full(K, snr_db, dtype=np.float32),
        np.full(K, delay_ms, dtype=np.float32),
        np.full(K, pilot_length, dtype=np.float32),
        run,
    )
    return [seq_k.astype(np.float32)] + cond


def evaluate_ofdm_wideband_extension(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    bundle: Dict[str, object],
) -> Dict[str, object]:
    """Shared-weight per-subcarrier wideband MIMO-OFDM evaluation.

    The returned diagnostics include the actual H[t,k] shape and measured
    variance across the subcarrier axis so a run can verify that frequency
    selectivity is real rather than decorative broadcasting.
    """
    methods = [
        "Comb-pilot LS interpolation",
        "Prop-Spatial shared per subcarrier",
        "Proposed physics-guided temporal shared per subcarrier",
    ]
    errs = {m: [] for m in methods}
    ici_penalty_db: List[float] = []
    freq_variances: List[float] = []
    observed_shapes: List[List[int]] = []
    n = min(run.ofdm_test_samples, run.num_test)

    for i in range(n):
        rng = np.random.default_rng(410_000 + i)
        times = np.arange(run.seq_len, dtype=float) * run.history_dt_s
        h = generate_effective_channel_trajectory_ofdm(
            cfg, rng, run.fixed_eval_speed_kmh / 3.6, times
        )
        observed_shapes.append(list(h.shape))
        hk = h[-1]
        centered = hk - np.mean(hk, axis=0, keepdims=True)
        freq_variances.append(float(np.mean(np.abs(centered) ** 2)))

        ls_hist = []
        for t in range(run.seq_len):
            _, ls_k = pilot_observation_and_ls_ofdm(
                h[t],
                run.fixed_eval_snr_db,
                run.main_pilot_length,
                run.ofdm_pilot_subcarrier_stride,
                rng,
            )
            ls_hist.append(ls_k)
        ls_hist = np.asarray(ls_hist)  # (T,K,Nr,Nt)
        true = h[-1]
        current_ls = ls_hist[-1]
        y = complex_to_image(true / scale)
        x = complex_to_image(current_ls / scale)
        prop_sp = bundle["spatial"].predict(x, verbose=0)
        seq_k = np.transpose(complex_to_image(ls_hist / scale), (1, 0, 2, 3, 4))
        tin = _replicate_temporal_conditions_for_subcarriers(
            seq_k,
            run.fixed_eval_speed_kmh,
            run.fixed_eval_snr_db,
            0.0,
            run.main_pilot_length,
            run,
        )
        prop_t = _predict_temporal(bundle["proposed"], tin)
        errs[methods[0]].append(nmse_linear(y, x))
        errs[methods[1]].append(nmse_linear(y, prop_sp))
        errs[methods[2]].append(nmse_linear(y, prop_t))

        fd = maximum_doppler_hz(run.fixed_eval_speed_kmh / 3.6, cfg.carrier_frequency_hz)
        ici_ratio = (np.pi * fd * cfg.ofdm_symbol_duration_s) ** 2 / 3.0
        ici_penalty_db.append(10.0 * np.log10(1.0 + ici_ratio))

    return {
        "nmse_db": {m: linear_to_db(float(np.mean(v))) for m, v in errs.items()},
        "mean_ici_snr_penalty_db_approx": float(np.mean(ici_penalty_db)),
        "mean_frequency_selectivity_variance": float(np.mean(freq_variances)),
        "minimum_frequency_selectivity_variance": float(np.min(freq_variances)),
        "frequency_selectivity_nonzero": bool(np.min(freq_variances) > 0.0),
        "observed_trajectory_shapes": observed_shapes,
        "expected_subcarrier_axis_size": int(cfg.num_subcarriers),
        "num_subcarriers": cfg.num_subcarriers,
        "num_taps": cfg.num_channel_taps,
        "pilot_subcarrier_stride": run.ofdm_pilot_subcarrier_stride,
        "nonpilot_estimation": "linear interpolation from noisy LS values at pilot subcarriers only; no true non-pilot H[k] is used",
        "scope_note": "shared-weight per-subcarrier extension; cross-frequency neural processing is not claimed",
    }



def evaluate_pathloss_mismatch(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    bundle: Dict[str, object],
) -> Dict[str, List[float]]:
    offsets = [0.0, 0.3, 0.6]
    out = {"pathloss_exponent_offset": offsets, "Prop-Spatial": [], "Proposed": [], "CNN-BiLSTM-inspired": []}
    for off in offsets:
        cfg_m = replace(
            cfg,
            pathloss_exp_bs_irs=cfg.pathloss_exp_bs_irs + off,
            pathloss_exp_irs_mcr=cfg.pathloss_exp_irs_mcr + off,
            pathloss_exp_direct=cfg.pathloss_exp_direct + off,
        )
        rng = np.random.default_rng(420_000 + int(off * 100))
        x, y, _, _ = build_temporal_dataset(
            cfg_m, run, min(run.num_test, 600), rng, scale,
            fixed_snr_db=run.fixed_eval_snr_db,
            fixed_speed_kmh=run.fixed_eval_speed_kmh,
            fixed_delay_ms=1.0,
            fixed_pilot_length=run.main_pilot_length,
        )
        out["Prop-Spatial"].append(nmse_db(y, bundle["spatial"].predict(x[0][:, -1], verbose=0)))
        out["Proposed"].append(nmse_db(y, _predict_temporal(bundle["proposed"], x)))
        out["CNN-BiLSTM-inspired"].append(nmse_db(y, _predict_temporal(bundle["cnn_bilstm"], x)))
    return out


def evaluate_cross_scenario_generalization(run: RunConfig) -> Dict[str, object]:
    scenarios = {
        "open-track/viaduct": SysConfig.open_track_viaduct(),
        "tunnel/cutting": SysConfig.tunnel_cutting(),
    }
    directions = [("open-track/viaduct", "tunnel/cutting"), ("tunnel/cutting", "open-track/viaduct")]
    result = {}
    for src_name, dst_name in directions:
        src, dst = scenarios[src_name], scenarios[dst_name]
        vals_cross, vals_same = [], []
        for seed in range(run.cross_scenario_seeds):
            local = replace(
                run,
                n_seeds=1,
                num_train_spatial=run.cross_scenario_train_samples,
                num_val_spatial=run.cross_scenario_val_samples,
                num_train_temporal=run.cross_scenario_train_samples,
                num_val_temporal=run.cross_scenario_val_samples,
                epochs_spatial=run.cross_scenario_epochs,
                epochs_temporal=run.cross_scenario_epochs,
                temporal_finetune_epochs=0,
                normalization_samples=min(run.normalization_samples, max(128, run.cross_scenario_train_samples // 2)),
            )
            scale_src = compute_global_channel_scale(src, local)
            sp_src = train_spatial_network(src, local, scale_src, 430_000 + seed, build_proposed_spatial)
            pg_src = train_proposed_temporal_physics_guided(src, local, scale_src, 431_000 + seed, sp_src)
            mu_src = models.Model(pg_src.inputs, layers.Lambda(lambda z: z[..., :2])(pg_src.output))

            scale_dst = compute_global_channel_scale(dst, local)
            sp_dst = train_spatial_network(dst, local, scale_dst, 432_000 + seed, build_proposed_spatial)
            pg_dst = train_proposed_temporal_physics_guided(dst, local, scale_dst, 433_000 + seed, sp_dst)
            mu_dst = models.Model(pg_dst.inputs, layers.Lambda(lambda z: z[..., :2])(pg_dst.output))

            rng = np.random.default_rng(434_000 + seed)
            # Cross-domain model retains source normalization by design.
            x_cross, y_cross, _, _ = build_temporal_dataset(
                dst, local, min(local.num_test, 400), rng, scale_src,
                fixed_snr_db=run.fixed_eval_snr_db,
                fixed_speed_kmh=run.fixed_eval_speed_kmh,
                fixed_delay_ms=1.0,
                fixed_pilot_length=run.main_pilot_length,
            )
            vals_cross.append(nmse_db(y_cross, mu_src.predict(x_cross, verbose=0)))
            x_same, y_same, _, _ = build_temporal_dataset(
                dst, local, min(local.num_test, 400), rng, scale_dst,
                fixed_snr_db=run.fixed_eval_snr_db,
                fixed_speed_kmh=run.fixed_eval_speed_kmh,
                fixed_delay_ms=1.0,
                fixed_pilot_length=run.main_pilot_length,
            )
            vals_same.append(nmse_db(y_same, mu_dst.predict(x_same, verbose=0)))
        result[f"{src_name} -> {dst_name}"] = {
            "cross_domain": summarize_seed_values(vals_cross),
            "same_domain_reference": summarize_seed_values(vals_same),
            "degradation_db": float(np.mean(vals_cross) - np.mean(vals_same)),
        }
    return result


def run_temporal_ablation(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    pretrained_spatial: tf.keras.Model,
) -> Dict[str, Dict[str, object]]:
    variants = {
        "Full physics-guided": {},
        "No physics guidance": {"use_physics_prior": False},
        "No speed conditioning": {"use_speed": False},
        "No SNR conditioning": {"use_snr": False},
        "No delay conditioning": {"use_delay": False},
        "No pilot-length conditioning": {"use_pilot": False},
        "No spatial front end": {"use_spatial_frontend": False},
        "No residual gate": {"use_gate": False},
        "Unidirectional LSTM": {"bidirectional": False},
        "Single recurrent layer": {"recurrent_layers": 1},
    }
    values = {k: [] for k in variants}

    for seed in range(run.ablation_seeds):
        rng = np.random.default_rng(440_000 + seed)
        x_test, y_test, _, _ = build_temporal_dataset(
            cfg,
            run,
            min(run.num_test, 600),
            rng,
            scale,
            fixed_snr_db=10.0,
            fixed_speed_kmh=400.0,
            fixed_delay_ms=1.0,
            fixed_pilot_length=run.main_pilot_length,
        )
        for idx, (name, variant) in enumerate(variants.items()):
            m = train_proposed_temporal_physics_guided(
                cfg,
                run,
                scale,
                441_000 + seed * 100 + idx,
                pretrained_spatial,
                variant=variant,
                num_train=run.ablation_train_samples,
                num_val=run.ablation_val_samples,
                epochs=run.ablation_epochs,
            )
            mu_model = models.Model(
                m.inputs, layers.Lambda(lambda z: z[..., :2])(m.output)
            )
            values[name].append(
                nmse_db(y_test, mu_model.predict(x_test, verbose=0))
            )

    out = {name: summarize_seed_values(v) for name, v in values.items()}
    full = float(out["Full physics-guided"]["mean"])
    for name, d in out.items():
        variant_nmse = float(d["mean"])
        full_gain = variant_nmse - full  # positive => full model is better
        d["full_model_gain_vs_variant_db"] = float(full_gain)
        d["absolute_ablation_effect_db"] = float(abs(full_gain))
        if name == "Full physics-guided":
            d["significance_flag"] = "REFERENCE"
        elif abs(full_gain) < 0.5:
            d["significance_flag"] = (
                "NOT SUBSTANTIVE - reconsider claiming this component"
            )
        elif full_gain > 0:
            d["significance_flag"] = "SUBSTANTIVE BENEFIT (>=0.5 dB)"
        else:
            d["significance_flag"] = (
                "SUBSTANTIVE NEGATIVE EFFECT - component hurts performance"
            )
    return out



def evaluate_irs_phase_strategy(cfg: SysConfig, run: RunConfig) -> Dict[str, float]:
    """Random vs phase-aligned vs phase-error-impaired IRS strategy diagnostic."""
    rng = np.random.default_rng(450_000)
    vals = {"Random IRS phase": [], "Phase-aligned optimized IRS": [], "Optimized + 10deg phase error": []}
    for i in range(min(run.irs_control_trajectories, 100)):
        comp = generate_effective_channel_components_trajectory(cfg, rng, run.fixed_eval_speed_kmh / 3.6, [0.0])
        hd, hr, g, phi0 = comp["h_d"][0], comp["h_r"][0], comp["g"], comp["phi_random"]
        h0 = effective_channel_from_components(hd, hr, g, phi0)
        phio = optimize_irs_phase(h0, hr, g, hd, run.irs_phase_iterations)
        phie = phio * np.exp(1j * np.deg2rad(10.0) * rng.standard_normal(len(phio)))
        for name, phi in [("Random IRS phase", phi0), ("Phase-aligned optimized IRS", phio), ("Optimized + 10deg phase error", phie)]:
            h = effective_channel_from_components(hd, hr, g, phi)
            vals[name].append(achievable_rate_physical_from_estimate(h, h, cfg))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def build_robustness_summary_table(all_results: Dict[str, object]) -> pd.DataFrame:
    """Build a computed reviewer-facing readiness table from executed results."""
    rows: List[Dict[str, object]] = []

    def add(condition, comparator, value, target, status):
        rows.append(
            {
                "Condition": condition,
                "Comparator": comparator,
                "Observed": value,
                "Target": target,
                "Status": status,
            }
        )

    nm = all_results.get("nmse_vs_snr_delayed", all_results.get("nmse_vs_snr", {}))
    prop_key = "Proposed physics-guided temporal"
    dl_names = [
        n
        for n in [
            "CDRN-inspired baseline",
            "CNN-BiLSTM-inspired baseline",
            "GRU-ODE-inspired temporal baseline",
            "MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)",
        ]
        if n in nm
    ]
    if prop_key in nm and dl_names:
        prop = np.asarray(nm[prop_key]["mean"], dtype=float)
        strongest = np.min(
            np.vstack([np.asarray(nm[n]["mean"], dtype=float) for n in dl_names]),
            axis=0,
        )
        gain = float(np.mean(strongest - prop))
        add(
            "Predictive NMSE vs SNR",
            "Strongest DL baseline",
            f"{gain:.2f} dB average gain",
            ">=1 dB",
            "PASS" if gain >= 1 else ("PARTIAL" if gain > 0 else "FAIL"),
        )

        # Statistical separation using the strongest baseline at each SNR.
        prop_ci = np.asarray(nm[prop_key].get("ci95", np.zeros_like(prop)), dtype=float)
        significant = []
        for i in range(len(prop)):
            best_name = min(dl_names, key=lambda n: float(nm[n]["mean"][i]))
            bmean = float(nm[best_name]["mean"][i])
            bci = float(nm[best_name].get("ci95", [0.0] * len(prop))[i])
            significant.append(bool(prop[i] + prop_ci[i] < bmean - bci))
        frac = float(np.mean(significant)) if significant else float("nan")
        add(
            "Five-seed statistical separation",
            "Strongest DL baseline with 95% CI",
            f"{100.0 * frac:.1f}% SNR points non-overlapping",
            "majority of points",
            "PASS" if np.isfinite(frac) and frac >= 0.5 else "PARTIAL",
        )

    aging = all_results.get("channel_aging", {})
    delays = np.asarray(all_results.get("aging_delays_ms", []), dtype=float)
    if (
        "Proposed physics-guided predictor" in aging
        and "Stale Matched LMMSE" in aging
        and delays.size
    ):
        prop_a = np.asarray(aging["Proposed physics-guided predictor"]["mean"], dtype=float)
        lm_a = np.asarray(aging["Stale Matched LMMSE"]["mean"], dtype=float)
        i0 = int(np.argmin(np.abs(delays - 0.0)))
        static_gap = float(prop_a[i0] - lm_a[i0])
        add(
            "Static / zero aging",
            "Matched LMMSE",
            f"{static_gap:+.2f} dB proposed-minus-LMMSE",
            "competitive (<=+1 dB)",
            "PASS" if static_gap <= 1.0 else "PARTIAL",
        )
        ids = np.where((delays >= 0.5) & (delays <= 3.0))[0]
        if len(ids):
            gain = float(np.mean(lm_a[ids] - prop_a[ids]))
            add(
                "0.5-3 ms CSI aging",
                "Stale matched LMMSE",
                f"{gain:.2f} dB gain",
                "2-5 dB",
                "PASS" if gain >= 2 else ("PARTIAL" if gain > 0 else "FAIL"),
            )

    speed = all_results.get("nmse_vs_speed", {})
    speed_points = np.asarray(all_results.get("speed_points_kmh", []), dtype=float)
    if prop_key in speed and speed_points.size:
        temporal_comp = [
            n
            for n in [
                "CNN-BiLSTM-inspired baseline",
                "GRU-ODE-inspired temporal baseline",
                "Temporal LMMSE (condition-matched)",
                "Frozen Temporal LMMSE (nominal 1 ms)",
                "Stale MultiScaleCNN GAN-inspired baseline",
                "Stale Prop-Spatial",
            ]
            if n in speed
        ]
        hi = np.where(speed_points >= 300)[0]
        if temporal_comp and len(hi):
            p = np.asarray(speed[prop_key]["mean"], dtype=float)[hi]
            b = np.min(
                np.vstack([
                    np.asarray(speed[n]["mean"], dtype=float)[hi]
                    for n in temporal_comp
                ]),
                axis=0,
            )
            gain = float(np.mean(b - p))
            add(
                "300-500 km/h mobility",
                "Strongest temporal/mobility baseline",
                f"{gain:.2f} dB average gain",
                ">0 and preferably increasing",
                "PASS" if gain > 0 else "FAIL",
            )
        held = np.where(speed_points > 400)[0]
        ref = np.where(speed_points == 400)[0]
        if len(held) and len(ref):
            p = np.asarray(speed[prop_key]["mean"], dtype=float)
            degradation = float(np.mean(p[held]) - p[ref[0]])
            add(
                "Held-out 450-500 km/h",
                "400 km/h in-distribution reference",
                f"{degradation:+.2f} dB degradation",
                "graceful (<3 dB)",
                "PASS" if degradation < 3 else "PARTIAL",
            )

    k = all_results.get("k_factor_mismatch", {})
    if "Proposed physics-guided temporal" in k and "Nominal LMMSE" in k:
        gain = float(k["Nominal LMMSE"][-1] - k["Proposed physics-guided temporal"][-1])
        add(
            "Severe K-factor mismatch",
            "Nominal/mismatched LMMSE",
            f"{gain:.2f} dB gain at strongest mismatch",
            ">0 dB",
            "PASS" if gain > 0 else "FAIL",
        )

    path = all_results.get("pathloss_mismatch", {})
    if "Proposed" in path and "CNN-BiLSTM-inspired" in path:
        gain = float(path["CNN-BiLSTM-inspired"][-1] - path["Proposed"][-1])
        add(
            "Path-loss mismatch",
            "CNN-BiLSTM-inspired baseline",
            f"{gain:.2f} dB gain at largest offset",
            ">0 dB",
            "PASS" if gain > 0 else "FAIL",
        )

    phase = all_results.get("phase_error_robustness", {})
    if "Proposed physics-guided predictor" in phase and "Stale Prop-Spatial" in phase:
        gain = float(phase["Stale Prop-Spatial"][-1] - phase["Proposed physics-guided predictor"][-1])
        add(
            "IRS phase impairment",
            "Stale Prop-Spatial",
            f"{gain:.2f} dB gain at maximum phase error",
            ">0 dB",
            "PASS" if gain > 0 else "FAIL",
        )

    closed = all_results.get("closed_loop_chain", {})
    if closed:
        cal = closed.get("uncertainty_calibration", {})
        corr = float(cal.get("uncertainty_error_correlation", np.nan))
        ece = float(cal.get("ece_rms", np.nan))
        c68 = float(cal.get("coverage_68", np.nan))
        c95 = float(cal.get("coverage_95", np.nan))
        add(
            "Uncertainty calibration",
            "Nominal probabilistic calibration",
            f"corr={corr:.2f}, ECE={ece:.3g}, cov68={c68:.2f}, cov95={c95:.2f}",
            "positive corr; low ECE; coverage near nominal",
            "PASS" if np.isfinite(corr) and corr > 0.3 else "PARTIAL",
        )
        ap = closed.get("adaptive_pilots", {}).get("headline", {})
        sav = float(ap.get("pilot_saving_percent_at_matched_nmse", np.nan))
        add(
            "Adaptive pilot efficiency",
            "Matched-NMSE fixed pilot policy",
            f"{sav:.1f}% signed saving",
            "25-50%",
            "PASS" if np.isfinite(sav) and sav >= 25 else ("PARTIAL" if np.isfinite(sav) and sav > 0 else "FAIL"),
        )
        ir = closed.get("predicted_csi_irs", {}).get("headline", {})
        rg = float(ir.get("predicted_minus_stale_rate_bps_hz", np.nan))
        add(
            "Predicted-CSI IRS optimization",
            "Stale-CSI optimized IRS",
            f"{rg:.3f} bit/s/Hz rate gain",
            ">0",
            "PASS" if np.isfinite(rg) and rg > 0 else "FAIL",
        )

    pilot = all_results.get("pilot_efficiency", {})
    net = pilot.get("net_spectral_efficiency", {}) if isinstance(pilot, dict) else {}
    pkey = "Proposed physics-guided temporal"
    if pkey in net:
        baseline_names = [n for n in net if n != pkey]
        if baseline_names:
            prop_best = float(np.max(net[pkey]))
            base_best = float(max(np.max(net[n]) for n in baseline_names))
            gain = prop_best - base_best
            add(
                "Net SE after pilot overhead",
                "Strongest fixed-pilot baseline",
                f"{gain:.3f} bit/s/Hz best-rate gain",
                ">0",
                "PASS" if gain > 0 else "FAIL",
            )

    ber = all_results.get("ber_vs_snr_delayed", all_results.get("ber_vs_snr", {}))
    if pkey in ber:
        baseline_names = [
            n for n in ber
            if n not in {pkey, "Perfect CSI"}
        ]
        if baseline_names:
            prop_curve = np.asarray(ber[pkey]["mean"], dtype=float)
            idx = len(prop_curve) // 2
            prop_ber = float(prop_curve[idx])
            best_base = float(
                min(np.asarray(ber[n]["mean"], dtype=float)[idx] for n in baseline_names)
            )
            add(
                "Predictive BER",
                "Strongest non-perfect-CSI baseline at mid-SNR",
                f"proposed={prop_ber:.3g}, baseline={best_base:.3g}",
                "lower than baseline",
                "PASS" if prop_ber < best_base else "PARTIAL",
            )

    cross = all_results.get("cross_scenario_generalization", {})
    if cross:
        deg = [v.get("degradation_db", np.nan) for v in cross.values()]
        mdeg = float(np.nanmean(deg))
        add(
            "Cross-scenario generalization",
            "Same-domain reference",
            f"{mdeg:.2f} dB mean degradation",
            "graceful (<5 dB)",
            "PASS" if mdeg < 5 else "PARTIAL",
        )

    complexity = all_results.get("complexity", {})
    if "Proposed physics-guided temporal" in complexity and "CNN-BiLSTM-inspired baseline" in complexity:
        p = complexity["Proposed physics-guided temporal"]
        b = complexity["CNN-BiLSTM-inspired baseline"]
        lat_ratio = float(p["inference_ms_per_sample"] / max(b["inference_ms_per_sample"], 1e-12))
        param_ratio = float(p["parameters"] / max(b["parameters"], 1))
        add(
            "Complexity",
            "CNN-BiLSTM-inspired baseline",
            f"latency x{lat_ratio:.2f}, parameters x{param_ratio:.2f}",
            "performance gain should justify overhead",
            "PASS" if lat_ratio <= 10.0 else "PARTIAL",
        )

    return pd.DataFrame(rows)



def save_robustness_summary_figure(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    code = {"PASS": 2, "PARTIAL": 1, "FAIL": 0}
    vals = np.array([[code.get(s, 1)] for s in df["Status"]], dtype=float)
    plt.figure(figsize=(10, max(4, 0.45 * len(df))))
    plt.imshow(vals, aspect="auto", vmin=0, vmax=2, cmap="RdYlGn")
    plt.yticks(np.arange(len(df)), df["Condition"])
    plt.xticks([0], ["Readiness status"])
    for i, row in df.iterrows():
        plt.text(0, i, f"{row['Status']}\n{row['Observed']}", ha="center", va="center", fontsize=8)
    plt.title("Computed robustness/readiness summary")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()



def evaluate_pilot_efficiency(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    bundle: Dict[str, object],
) -> Dict[str, Dict[str, List[float]]]:
    methods = [
        "Stale LS",
        "Stale Matched LMMSE",
        "Stale Prop-Spatial",
        "Temporal LMMSE",
        "CNN-BiLSTM-inspired baseline",
        "Proposed physics-guided temporal",
    ]
    if run.include_gan_baseline:
        methods.insert(3, "Stale MultiScaleCNN GAN-inspired baseline")
    if run.include_ode_baseline:
        methods.insert(-1, "GRU-ODE-inspired temporal baseline")
    nmse = {m: [] for m in methods}
    net_rate = {m: [] for m in methods}
    coh_symbols = coherence_symbols(cfg, run.fixed_eval_speed_kmh)

    for tau in run.pilot_lengths:
        rng = np.random.default_rng(40_000 + tau)
        x_t, y_t, _, meta = build_temporal_dataset(
            cfg,
            run,
            min(run.num_test, 600),
            rng,
            scale,
            fixed_snr_db=run.fixed_eval_snr_db,
            fixed_speed_kmh=run.fixed_eval_speed_kmh,
            fixed_delay_ms=run.pilot_eval_prediction_delay_ms,
            fixed_pilot_length=tau,
            return_raw=True,
        )
        assert meta is not None
        target_raw = meta["raw_target"]
        current_ls = x_t[0][:, -1]
        spatial_lmmse = _fit_spatial_lmmse_for_condition(
            cfg, run, scale, rng, run.fixed_eval_snr_db, run.fixed_eval_speed_kmh, tau
        )
        temporal_lmmse = _fit_temporal_lmmse_for_condition(
            cfg,
            run,
            scale,
            rng,
            run.fixed_eval_snr_db,
            run.fixed_eval_speed_kmh,
            run.pilot_eval_prediction_delay_ms,
            tau,
        )
        preds = {
            "Stale LS": current_ls,
            "Stale Matched LMMSE": spatial_lmmse.predict(current_ls),
            "Stale Prop-Spatial": bundle["spatial"].predict(current_ls, verbose=0),
            "Temporal LMMSE": temporal_lmmse.predict(x_t[0]),
            "CNN-BiLSTM-inspired baseline": _predict_temporal(bundle["cnn_bilstm"], x_t),
            "Proposed physics-guided temporal": _predict_temporal(bundle["proposed"], x_t),
        }
        if run.include_gan_baseline:
            preds["Stale MultiScaleCNN GAN-inspired baseline"] = bundle["multiscale"].predict(
                current_ls, verbose=0
            )
        if run.include_ode_baseline:
            preds["GRU-ODE-inspired temporal baseline"] = _predict_temporal(
                bundle["ode_rnn"], x_t
            )

        overhead = max(0.0, 1.0 - tau / max(coh_symbols, 1))
        for m, pred_norm in preds.items():
            nmse[m].append(nmse_db(y_t, pred_norm))
            rates = []
            for i in range(len(target_raw)):
                h_est_raw = image_to_complex(pred_norm[i]) * scale
                rates.append(
                    achievable_rate_physical_from_estimate(
                        target_raw[i], h_est_raw, cfg
                    )
                )
            net_rate[m].append(float(np.mean(rates) * overhead))

    return {
        "nmse": nmse,
        "net_spectral_efficiency": net_rate,
        "coherence_symbols": {"value": [float(coh_symbols)]},
        "unseen_pilot_length_generalization": [
            int(t)
            for t in run.pilot_lengths
            if int(t) not in set(run.training_pilot_lengths)
        ],
    }



def evaluate_ber_and_rate_vs_snr(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    models_by_seed: List[Dict[str, object]],
    *,
    delay_ms: float = 0.0,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    # Five-seed BER and spectral-efficiency evaluation at a chosen CSI delay.
    methods = [
        "Perfect CSI",
        "Stale Matched LMMSE",
        "Temporal LMMSE (condition-matched)",
        "Frozen Temporal LMMSE (nominal 1 ms)",
        "CDRN-inspired baseline",
        "Prop-Spatial",
        "CNN-BiLSTM-inspired baseline",
        "Proposed physics-guided temporal",
    ]
    if run.include_gan_baseline:
        methods.insert(-3, "MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)")
    if run.include_ode_baseline:
        methods.insert(-1, "GRU-ODE-inspired temporal baseline")

    ber_seed = {m: {float(s): [] for s in run.snr_points_db} for m in methods}
    rate_seed = {m: {float(s): [] for s in run.snr_points_db} for m in methods}

    for bundle in models_by_seed:
        seed = int(bundle["seed"])

        rng_frozen = np.random.default_rng(seed + 49_500)
        frozen_tl = _fit_temporal_lmmse_for_condition(
            cfg, run, scale, rng_frozen,
            run.fixed_eval_snr_db,
            run.fixed_eval_speed_kmh,
            run.headline_prediction_delay_ms,
            run.main_pilot_length,
        )

        for snr in run.snr_points_db:
            rng = np.random.default_rng(
                seed + 50_000 + int((snr + 20) * 10) + int(delay_ms * 1000)
            )
            nchan = min(run.ber_channel_samples, run.num_test)

            x_t, _, _, meta = build_temporal_dataset(
                cfg, run, nchan, rng, scale,
                fixed_snr_db=snr,
                fixed_speed_kmh=run.fixed_eval_speed_kmh,
                fixed_delay_ms=delay_ms,
                fixed_pilot_length=run.main_pilot_length,
                return_raw=True,
            )
            assert meta is not None
            h_true_raw = meta["raw_target"]
            current_ls = x_t[0][:, -1]

            spatial_lmmse = _fit_spatial_lmmse_for_condition(
                cfg, run, scale, rng,
                snr, run.fixed_eval_speed_kmh, run.main_pilot_length,
            )
            temporal_lmmse = _fit_temporal_lmmse_for_condition(
                cfg, run, scale, rng,
                snr, run.fixed_eval_speed_kmh, delay_ms, run.main_pilot_length,
            )

            preds = {
                "Stale Matched LMMSE": spatial_lmmse.predict(current_ls),
                "Temporal LMMSE (condition-matched)": temporal_lmmse.predict(x_t[0]),
                "Frozen Temporal LMMSE (nominal 1 ms)": frozen_tl.predict(x_t[0]),
                "CDRN-inspired baseline": bundle["cdrn"].predict(current_ls, verbose=0),
                "Prop-Spatial": bundle["spatial"].predict(current_ls, verbose=0),
                "CNN-BiLSTM-inspired baseline": _predict_temporal(bundle["cnn_bilstm"], x_t),
                "Proposed physics-guided temporal": _predict_temporal(bundle["proposed"], x_t),
            }
            if run.include_gan_baseline:
                preds["MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)"] = (
                    bundle["multiscale"].predict(current_ls, verbose=0)
                )
            if run.include_ode_baseline:
                preds["GRU-ODE-inspired temporal baseline"] = _predict_temporal(
                    bundle["ode_rnn"], x_t
                )

            err_counts = {m: 0 for m in methods}
            bit_counts = {m: 0 for m in methods}
            rate_values = {m: [] for m in methods}

            for i in range(nchan):
                h_true_norm = h_true_raw[i] / scale

                e, b = qpsk_ber_one_channel(
                    h_true_norm, h_true_norm, snr,
                    run.ber_symbols_per_channel, rng, perfect=True
                )
                err_counts["Perfect CSI"] += e
                bit_counts["Perfect CSI"] += b
                rate_values["Perfect CSI"].append(
                    achievable_rate_perfect(h_true_norm, snr)
                )

                for m, pred in preds.items():
                    h_est_norm = image_to_complex(pred[i])
                    e, b = qpsk_ber_one_channel(
                        h_true_norm, h_est_norm, snr,
                        run.ber_symbols_per_channel, rng
                    )
                    err_counts[m] += e
                    bit_counts[m] += b
                    rate_values[m].append(
                        achievable_rate_from_estimate(h_true_norm, h_est_norm, snr)
                    )

            for m in methods:
                ber_seed[m][float(snr)].append(
                    float(err_counts[m] / max(1, bit_counts[m]))
                )
                rate_seed[m][float(snr)].append(
                    float(np.mean(rate_values[m]))
                )

    def summarize(grid):
        out = {}
        for m in methods:
            stats = [summarize_seed_values(grid[m][float(s)]) for s in run.snr_points_db]
            out[m] = {
                "mean": [z["mean"] for z in stats],
                "std": [z["std"] for z in stats],
                "ci95": [z["ci95"] for z in stats],
            }
        return out

    return summarize(ber_seed), summarize(rate_seed)

def evaluate_k_mismatch(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    bundle: Dict[str, object],
) -> Dict[str, List[float]]:
    methods = ["Oracle Matched LMMSE", "Nominal LMMSE", "CDRN-inspired baseline", "Prop-Spatial", "CNN-BiLSTM-inspired baseline", "Proposed physics-guided temporal"]
    if run.include_gan_baseline: methods.insert(4, "MultiScaleCNN GAN-inspired baseline")
    if run.include_ode_baseline: methods.insert(-1, "GRU-ODE-inspired temporal baseline")
    out = {m: [] for m in methods}
    rng_nom = np.random.default_rng(60_000)
    nominal_lmmse = _fit_spatial_lmmse_for_condition(
        cfg, run, scale, rng_nom, run.fixed_eval_snr_db,
        run.fixed_eval_speed_kmh, run.main_pilot_length, k_offset_db=0.0,
    )
    for ko in run.k_offsets_db:
        rng = np.random.default_rng(60_100 + int(abs(ko) * 10))
        x_t, y_t, _, _ = build_temporal_dataset(
            cfg, run, min(run.num_test, 600), rng, scale,
            fixed_snr_db=run.fixed_eval_snr_db,
            fixed_speed_kmh=run.fixed_eval_speed_kmh,
            fixed_delay_ms=0.0,
            fixed_pilot_length=run.main_pilot_length,
            k_offset_db=ko,
        )
        cur = x_t[0][:, -1]
        oracle = _fit_spatial_lmmse_for_condition(
            cfg, run, scale, rng, run.fixed_eval_snr_db,
            run.fixed_eval_speed_kmh, run.main_pilot_length, k_offset_db=ko,
        )
        out["Oracle Matched LMMSE"].append(nmse_db(y_t, oracle.predict(cur)))
        out["Nominal LMMSE"].append(nmse_db(y_t, nominal_lmmse.predict(cur)))
        out["CDRN-inspired baseline"].append(nmse_db(y_t, bundle["cdrn"].predict(cur, verbose=0)))
        out["Prop-Spatial"].append(nmse_db(y_t, bundle["spatial"].predict(cur, verbose=0)))
        out["CNN-BiLSTM-inspired baseline"].append(nmse_db(y_t, _predict_temporal(bundle["cnn_bilstm"], x_t)))
        out["Proposed physics-guided temporal"].append(nmse_db(y_t, _predict_temporal(bundle["proposed"], x_t)))
        if run.include_gan_baseline: out["MultiScaleCNN GAN-inspired baseline"].append(nmse_db(y_t, bundle["multiscale"].predict(cur, verbose=0)))
        if run.include_ode_baseline: out["GRU-ODE-inspired temporal baseline"].append(nmse_db(y_t, _predict_temporal(bundle["ode_rnn"], x_t)))
    return out


def evaluate_phase_error_robustness(
    cfg: SysConfig,
    run: RunConfig,
    scale: float,
    bundle: Dict[str, object],
) -> Dict[str, List[float]]:
    methods = ["Stale LS", "Stale Prop-Spatial", "CNN-BiLSTM-inspired predictor", "Proposed physics-guided predictor"]
    if run.include_ode_baseline: methods.insert(-1, "GRU-ODE-inspired predictor")
    out = {m: [] for m in methods}
    for pe in run.phase_error_std_deg:
        rng = np.random.default_rng(61_000 + int(pe * 10))
        x_t, y_t, _, _ = build_temporal_dataset(
            cfg, run, min(run.num_test, 600), rng, scale,
            fixed_snr_db=run.fixed_eval_snr_db,
            fixed_speed_kmh=run.fixed_eval_speed_kmh,
            fixed_delay_ms=run.pilot_eval_prediction_delay_ms,
            fixed_pilot_length=run.main_pilot_length,
            phase_error_std_deg=pe,
        )
        cur = x_t[0][:, -1]
        out["Stale LS"].append(nmse_db(y_t, cur))
        out["Stale Prop-Spatial"].append(nmse_db(y_t, bundle["spatial"].predict(cur, verbose=0)))
        out["CNN-BiLSTM-inspired predictor"].append(nmse_db(y_t, _predict_temporal(bundle["cnn_bilstm"], x_t)))
        out["Proposed physics-guided predictor"].append(nmse_db(y_t, _predict_temporal(bundle["proposed"], x_t)))
        if run.include_ode_baseline: out["GRU-ODE-inspired predictor"].append(nmse_db(y_t, _predict_temporal(bundle["ode_rnn"], x_t)))
    return out


def evaluate_irs_scalability(run: RunConfig) -> Dict[str, object]:
    results = {m: {irs: [] for irs in run.irs_sizes} for m in ["LS", "Matched LMMSE", "Prop-Spatial", "Proposed physics-guided temporal", "Raw channel power (dB)"]}
    for irs in run.irs_sizes:
        cfg_m = SysConfig(num_irs_elements=irs)
        local = replace(
            run,
            num_train_spatial=run.irs_sweep_train_samples,
            num_val_spatial=run.irs_sweep_val_samples,
            num_train_temporal=run.irs_sweep_train_samples,
            num_val_temporal=run.irs_sweep_val_samples,
            epochs_spatial=run.irs_sweep_epochs,
            epochs_temporal=run.irs_sweep_epochs,
            temporal_finetune_epochs=0,
            normalization_samples=min(run.normalization_samples, max(256, run.irs_sweep_train_samples // 2)),
        )
        scale_m = compute_global_channel_scale(cfg_m, local)
        for seed in range(run.irs_sweep_seeds):
            sp = train_spatial_network(cfg_m, local, scale_m, 70_000 + irs + seed, build_proposed_spatial)
            pg = train_proposed_temporal_physics_guided(cfg_m, local, scale_m, 71_000 + irs + seed, sp)
            mu_pg = models.Model(pg.inputs, layers.Lambda(lambda z: z[..., :2])(pg.output))
            rng = np.random.default_rng(72_000 + irs + seed)
            x_t, y_t, _, meta = build_temporal_dataset(
                cfg_m, local, run.irs_sweep_test_samples, rng, scale_m,
                fixed_snr_db=run.fixed_eval_snr_db,
                fixed_speed_kmh=run.fixed_eval_speed_kmh,
                fixed_delay_ms=0.0,
                fixed_pilot_length=run.main_pilot_length,
                return_raw=True,
            )
            assert meta is not None
            cur = x_t[0][:, -1]
            lm = _fit_spatial_lmmse_for_condition(cfg_m, local, scale_m, rng, run.fixed_eval_snr_db, run.fixed_eval_speed_kmh, run.main_pilot_length)
            results["LS"][irs].append(nmse_db(y_t, cur))
            results["Matched LMMSE"][irs].append(nmse_db(y_t, lm.predict(cur)))
            results["Prop-Spatial"][irs].append(nmse_db(y_t, sp.predict(cur, verbose=0)))
            results["Proposed physics-guided temporal"][irs].append(nmse_db(y_t, mu_pg.predict(x_t, verbose=0)))
            results["Raw channel power (dB)"][irs].append(linear_to_db(float(np.mean(np.abs(meta["raw_target"]) ** 2))))
    out = {}
    for m, by in results.items():
        stats = [summarize_seed_values(by[irs]) for irs in run.irs_sizes]
        out[m] = {"mean": [z["mean"] for z in stats], "std": [z["std"] for z in stats], "ci95": [z["ci95"] for z in stats]}
    return out

# =============================================================================
# Plot and export helpers
# =============================================================================


_DISTINCT_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p"]
_DISTINCT_LINESTYLES = ["-", "--", "-.", ":"]


def plot_curves_with_ci(
    x: Sequence[float],
    summary: Dict[str, Dict[str, List[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    semilogy: bool = False,
) -> None:
    """Plot mean +/- 95% CI curves for every method in ``summary``.

    V6 fix: matplotlib's default color cycle only has 10 colors. With 11+
    methods (as in the main NMSE-vs-SNR/aging plots), two curves silently
    reused the same color and became visually indistinguishable -- most
    notably "LS" and "Proposed physics-guided temporal", which made it
    impossible to tell from the figure alone whether the proposed model or
    LS was responsible for a high-SNR NMSE floor. We now draw from a
    colormap sized to the actual number of series and additionally cycle
    marker and linestyle, so no two methods can render identically even if a
    palette happens to produce two similar colors.
    """
    plt.figure(figsize=(9, 6))
    plottable = [
        (method, data) for method, data in summary.items()
        if "mean" in data and len(data["mean"]) == len(x)
    ]
    n = max(len(plottable), 1)
    colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, n))
    for idx, (method, data) in enumerate(plottable):
        y = np.asarray(data["mean"], dtype=float)
        ci = np.asarray(data.get("ci95", np.zeros_like(y)), dtype=float)
        color = colors[idx]
        marker = _DISTINCT_MARKERS[idx % len(_DISTINCT_MARKERS)]
        linestyle = _DISTINCT_LINESTYLES[(idx // len(_DISTINCT_MARKERS)) % len(_DISTINCT_LINESTYLES)]
        if semilogy:
            plt.semilogy(x, y, marker=marker, linestyle=linestyle, color=color, label=method)
        else:
            plt.plot(x, y, marker=marker, linestyle=linestyle, color=color, label=method)
            if np.any(ci > 0):
                plt.fill_between(x, y - ci, y + ci, color=color, alpha=0.12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_dict_table(path: Path, x_name: str, x: Sequence[float], curves: Dict[str, Sequence[float]]) -> None:
    data = {x_name: list(x)}
    for k, v in curves.items():
        data[k] = list(v)
    pd.DataFrame(data).to_csv(path, index=False)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# =============================================================================
# Full experiment runner
# =============================================================================


def run_pipeline(cfg: SysConfig, run: RunConfig) -> None:
    out_dir = Path(run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_seed_bundle(seed: int, bundle: Dict[str, object]) -> None:
        """Save every trained Keras model for this seed to disk immediately
        after it finishes. Training (not evaluation) is >90% of total
        runtime, so this is the single highest-value checkpoint: a
        disconnect after seed 3 of 5 no longer forces retraining seeds 0-2."""
        seed_dir = out_dir / f"seed_{seed}_models"
        seed_dir.mkdir(parents=True, exist_ok=True)
        required_keys = {"spatial", "cdrn", "cnn_bilstm", "proposed_prob"}
        if run.include_gan_baseline:
            required_keys.add("multiscale")
        if run.include_ode_baseline:
            required_keys.add("ode_rnn")

        saved_keys = set()
        save_failed = False
        for key, obj in bundle.items():
            if key == "seed":
                continue
            if key == "proposed":
                continue
            try:
                obj.save(seed_dir / f"{key}.keras")
                saved_keys.add(key)
            except Exception as e:
                save_failed = True
                print(f"  [warning: could not save {key} for seed {seed}: {e}]")

        missing = sorted(required_keys - saved_keys)
        if save_failed or missing:
            flag = seed_dir / "_complete.flag"
            if flag.exists():
                flag.unlink()
            raise RuntimeError(
                f"Seed {seed} checkpoint incomplete; missing/failed models: {missing}"
            )

        (seed_dir / "_complete.flag").write_text("done")
        print(f"  [seed {seed} models saved to {seed_dir}]")

    def try_load_seed_bundle(seed: int) -> Optional[Dict[str, object]]:
        """Return a previously-completed seed's bundle from disk, or None
        if this seed hasn't been fully trained+saved yet (e.g. first run,
        or the run died mid-seed before the completion flag was written)."""
        seed_dir = out_dir / f"seed_{seed}_models"
        if not (seed_dir / "_complete.flag").exists():
            return None
        bundle: Dict[str, object] = {"seed": seed}
        for key in ["spatial", "cdrn", "cnn_bilstm", "proposed_prob", "multiscale", "ode_rnn"]:
            path = seed_dir / f"{key}.keras"
            if path.exists():
                bundle[key] = tf.keras.models.load_model(path, safe_mode=False)
        if "proposed_prob" in bundle:
            proposed_prob = bundle["proposed_prob"]
            bundle["proposed"] = models.Model(
                proposed_prob.inputs,
                layers.Lambda(lambda z: z[..., :2], name=f"proposed_mu_seed_{seed}_resumed")(proposed_prob.output),
                name=f"ProposedPhysicsGuidedMean_seed{seed}_resumed",
            )
        print(f"  [seed {seed} loaded from checkpoint, skipping retraining]")
        return bundle

    def checkpoint(name: str, data: object) -> None:
        """Write intermediate results immediately, so a disconnect/crash
        partway through the (multi-hour) publication run doesn't lose
        everything computed up to that point -- only whatever came after
        the last checkpoint. Safe to call repeatedly; each call overwrites
        its own file only."""
        path = out_dir / f"checkpoint_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_safe(data), f, indent=2)
        print(f"  [checkpoint saved: {path.name}]")

    print("=== Rohitash Transactions-Ready Publication Pipeline v6 ===")
    print(f"Output: {out_dir.resolve()}")
    print("Primary framing: mobility-conditioned predictive effective-channel estimation with physics guidance, uncertainty, adaptive pilots, and predicted-CSI IRS control.")
    print("Computing fixed global normalization scale...")
    scale = compute_global_channel_scale(cfg, run)
    print(f"Global channel RMS scale = {scale:.6e}")

    metadata = {
        "system_config": asdict(cfg),
        "run_config": asdict(run),
        "global_channel_scale": scale,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "scientific_notes": {
            "pilot_model": "Yp = H_eff Xp + N with orthogonal DFT pilots and LS initialization",
            "normalization": "single fixed RMS scale computed from an independent training-realization pool",
            "channel_scope": "effective-channel estimation/prediction; H_d, H_r, and G are not claimed as individually recovered",
            "temporal_evolution": "Jakes-correlation-guided temporal channel evolution with incrementally accumulated Doppler phase",
            "proposed_model": "signed effective-Doppler Jakes-correlation prior + learned gated residual + heteroscedastic uncertainty",
            "temporal_fairness": "CNN-BiLSTM-inspired, GRU-ODE-inspired, temporal LMMSE, and proposed temporal models receive the same LS-history sequence",
            "baseline_scope": "CDRN-inspired and CNN-BiLSTM-inspired labels denote architecture-level baselines, not exact literature reproductions",
            "multiscale_scope": "GAN-inspired multi-scale CNN is non-adversarial and is not claimed as an exact reproduction of the 2026 RIS-HSR GAN paper",
            "aging_fairness": "all methods are evaluated against the target from the same continuous trajectory realization",
            "jakes_reference": "correlation reference only; not claimed as a standardized Jakes fading realization or a strict bound",
            "training_holdout": f"speed > {run.speed_train_max_kmh} km/h, delay > {run.delay_train_max_ms} ms, and pilot lengths outside {run.training_pilot_lengths} are extrapolation tests",
            "ofdm_extension": "stylized wideband effective-channel extension with multipath taps and shared per-subcarrier estimator weights; cross-frequency neural processing is not claimed",
            "no_result_fabrication": "all reported quantities are computed from executed simulations/models; no target NMSE values are hard-coded",
            "adaptive_pilot_streaming": "historical pilot observations are retained; only each new observation uses the newly selected tau",
            "adaptive_pilot_claim_scope": "diagnostic only unless the executed run beats fixed-pilot references",
            "irs_control_claim_scope": "diagnostic only unless the executed run beats stale/random IRS controls",
            "ofdm_irs_consistency": "all multipath taps share one common IRS phase state and common BS-IRS channel",
            "physics_ablation_integrity": "speed/SNR/delay/pilot masks are applied consistently to learned and physics paths",
        },
    }
    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)

    # ------------------------------------------------------------------
    # Train principal models across seeds.
    # ------------------------------------------------------------------
    models_by_seed: List[Dict[str, object]] = []
    for seed in range(run.n_seeds):
        resumed = try_load_seed_bundle(seed)
        if resumed is not None:
            models_by_seed.append(resumed)
            continue

        print(f"\n[Seed {seed}] Training Prop-Spatial...")
        spatial = train_spatial_network(cfg, run, scale, seed, build_proposed_spatial)
        print(f"[Seed {seed}] Training CDRN-inspired baseline...")
        cdrn = train_spatial_network(cfg, run, scale, seed + 100, build_cdrn_baseline)

        multiscale = None
        if run.include_gan_baseline:
            print(f"[Seed {seed}] Training MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)...")
            multiscale = train_multiscale_cnn_baseline(cfg, run, scale, seed)

        print(f"[Seed {seed}] Training CNN-BiLSTM-inspired baseline...")
        cnn_bilstm = train_cnn_bilstm_baseline(cfg, run, scale, seed)

        ode_rnn = None
        if run.include_ode_baseline:
            print(f"[Seed {seed}] Training GRU-ODE-inspired temporal baseline...")
            ode_rnn = train_ode_rnn_baseline(cfg, run, scale, seed)

        print(f"[Seed {seed}] Training proposed physics-guided uncertainty-aware predictor...")
        proposed_prob = train_proposed_temporal_physics_guided(cfg, run, scale, seed, spatial)
        proposed_mu = models.Model(
            proposed_prob.inputs,
            layers.Lambda(lambda z: z[..., :2], name=f"proposed_mu_seed_{seed}")(proposed_prob.output),
            name=f"ProposedPhysicsGuidedMean_seed{seed}",
        )

        bundle = {
            "seed": seed,
            "spatial": spatial,
            "cdrn": cdrn,
            "cnn_bilstm": cnn_bilstm,
            "proposed_prob": proposed_prob,
            "proposed": proposed_mu,
        }
        if multiscale is not None:
            bundle["multiscale"] = multiscale
        if ode_rnn is not None:
            bundle["ode_rnn"] = ode_rnn
        save_seed_bundle(seed, bundle)
        models_by_seed.append(bundle)

    primary = models_by_seed[0]

    # ------------------------------------------------------------------
    # 1. Main NMSE vs SNR with complete baseline roster.
    # ------------------------------------------------------------------
    print("\nEvaluating NMSE vs SNR with full baseline roster...")
    nmse_snr = evaluate_nmse_vs_snr(cfg, run, scale, models_by_seed)
    checkpoint("01_nmse_snr", nmse_snr)
    plot_curves_with_ci(
        run.snr_points_db, nmse_snr, "SNR (dB)", "NMSE (dB)",
        f"Effective-channel NMSE vs SNR ({run.n_seeds}-seed mean ± 95% CI)",
        out_dir / "01_nmse_vs_snr.png",
    )
    spatial_names = [n for n in [
        "LS", "Rank-1 SVD baseline", "Matched LMMSE", "Mismatched LMMSE",
        "CDRN-inspired baseline", "MultiScaleCNN baseline (GAN-inspired, non-adversarial reproduction)",
        "Prop-Spatial",
    ] if n in nmse_snr]
    temporal_names = [n for n in [
        "Temporal LMMSE", "CNN-BiLSTM-inspired baseline",
        "GRU-ODE-inspired temporal baseline", "Proposed physics-guided temporal",
    ] if n in nmse_snr]
    plot_curves_with_ci(
        run.snr_points_db, {k: nmse_snr[k] for k in spatial_names},
        "SNR (dB)", "NMSE (dB)", "Spatial estimators: one current pilot block",
        out_dir / "01a_nmse_spatial_fair.png",
    )
    plot_curves_with_ci(
        run.snr_points_db, {k: nmse_snr[k] for k in temporal_names},
        "SNR (dB)", "NMSE (dB)", "Temporal estimators: equal four-snapshot pilot history",
        out_dir / "01b_nmse_temporal_fair.png",
    )

    # ------------------------------------------------------------------
    # 1c. NMSE vs SNR at a genuine prediction horizon (delay > 0), so the
    #     temporal comparison is not dominated by the delay=0 identity case.
    # ------------------------------------------------------------------
    print(f"\nEvaluating NMSE vs SNR at a {run.headline_prediction_delay_ms} ms prediction horizon...")
    nmse_snr_delayed = evaluate_nmse_vs_snr(
        cfg, run, scale, models_by_seed,
        temporal_delay_ms=run.headline_prediction_delay_ms,
        include_spatial=False,
    )
    checkpoint("01c_nmse_snr_delayed", nmse_snr_delayed)
    plot_curves_with_ci(
        run.snr_points_db, nmse_snr_delayed, "SNR (dB)", "NMSE (dB)",
        f"Temporal NMSE vs SNR at {run.headline_prediction_delay_ms} ms prediction horizon "
        f"({run.n_seeds}-seed mean ± 95% CI)",
        out_dir / "01c_nmse_vs_snr_delayed.png",
    )

    # ------------------------------------------------------------------
    # 2. Speed generalization with held-out region explicitly marked.
    # ------------------------------------------------------------------
    print("Evaluating train-speed generalization...")
    speed_result = evaluate_nmse_vs_speed(cfg, run, scale, models_by_seed)
    checkpoint("02_nmse_speed", speed_result)
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, data in speed_result.items():
        ax.plot(run.speed_points_kmh, data["mean"], marker="o", label=method)
        ci = np.asarray(data.get("ci95", np.zeros(len(run.speed_points_kmh))))
        if np.any(ci > 0):
            y = np.asarray(data["mean"]); ax.fill_between(run.speed_points_kmh, y-ci, y+ci, alpha=0.10)
    ax.axvspan(run.speed_train_max_kmh, max(run.speed_points_kmh), alpha=0.08, label="held-out / extrapolation region")
    ax.axvline(run.speed_train_max_kmh, linestyle="--", linewidth=1)
    ax.set_xlabel("Train speed (km/h)"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Mobility robustness (prediction horizon={run.speed_eval_prediction_delay_ms} ms)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "02_nmse_vs_speed.png", dpi=250); plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Same-trajectory channel aging with held-out delay region.
    # ------------------------------------------------------------------
    print("Evaluating channel aging on common trajectories...")
    aging_result = evaluate_channel_aging(cfg, run, scale, models_by_seed)
    checkpoint("03_channel_aging", aging_result)
    fig, ax = plt.subplots(figsize=(9, 6))
    aging_colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, max(len(aging_result), 1)))
    for idx, (method, data) in enumerate(aging_result.items()):
        ax.plot(
            run.aging_delays_ms, data["mean"],
            marker=_DISTINCT_MARKERS[idx % len(_DISTINCT_MARKERS)],
            linestyle=_DISTINCT_LINESTYLES[(idx // len(_DISTINCT_MARKERS)) % len(_DISTINCT_LINESTYLES)],
            color=aging_colors[idx],
            label=method,
        )
    ax.axvspan(run.delay_train_max_ms, max(run.aging_delays_ms), alpha=0.08, label="held-out / extrapolation region")
    ax.axvline(run.delay_train_max_ms, linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction / CSI aging delay (ms)"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Channel aging at {run.fixed_eval_speed_kmh:.0f} km/h, {run.fixed_eval_snr_db:.0f} dB")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "03_channel_aging.png", dpi=250); plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Fixed pilot lengths + net SE; unseen tau points marked.
    # ------------------------------------------------------------------
    print("Evaluating pilot length and pilot-overhead-aware spectral efficiency...")
    pilot_result = evaluate_pilot_efficiency(cfg, run, scale, primary)
    checkpoint("04_pilot_efficiency", pilot_result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for m, vals in pilot_result["nmse"].items():
        axes[0].plot(run.pilot_lengths, vals, marker="o", label=m)
    unseen_tau = pilot_result.get("unseen_pilot_length_generalization", [])
    for tau in unseen_tau:
        axes[0].axvline(tau, linestyle=":", linewidth=1)
        axes[1].axvline(tau, linestyle=":", linewidth=1)
    axes[0].set_xlabel("Orthogonal pilot length τ"); axes[0].set_ylabel("NMSE (dB)")
    axes[0].set_title("Pilot efficiency (dotted = unseen pilot-length generalization)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=7)
    for m, vals in pilot_result["net_spectral_efficiency"].items():
        axes[1].plot(run.pilot_lengths, vals, marker="o", label=m)
    axes[1].set_xlabel("Orthogonal pilot length τ"); axes[1].set_ylabel("Net spectral efficiency (bit/s/Hz)")
    axes[1].set_title("Pilot-overhead-aware rate")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_dir / "04_pilot_efficiency_and_net_se.png", dpi=250); plt.close()

    # ------------------------------------------------------------------
    # 5. Priority-0 closed loop: uncertainty -> adaptive pilots -> future IRS.
    # ------------------------------------------------------------------
    print("Evaluating uncertainty calibration, adaptive pilots, and predicted-CSI IRS control...")
    closed_loop = evaluate_closed_loop_chain(cfg, run, scale, primary["proposed_prob"], out_dir=out_dir)

    # ------------------------------------------------------------------
    # 6. IRS scalability and raw physical channel power.
    # ------------------------------------------------------------------
    print("Evaluating IRS-size scalability (retraining per M)...")
    irs_result = evaluate_irs_scalability(run)
    checkpoint("06_irs_scalability", irs_result)
    nmse_irs = {k: v for k, v in irs_result.items() if k != "Raw channel power (dB)"}
    plot_curves_with_ci(
        run.irs_sizes, nmse_irs, "IRS elements M", "NMSE (dB)",
        "IRS-size scalability", out_dir / "05_nmse_vs_irs_size.png",
    )
    pwr = irs_result["Raw channel power (dB)"]
    plt.figure(figsize=(8, 5.5)); plt.plot(run.irs_sizes, pwr["mean"], marker="o")
    plt.xlabel("IRS elements M"); plt.ylabel("Average raw effective-channel power (dB)")
    plt.title("Physical array/path-gain effect before normalization"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "05b_raw_channel_power_vs_irs_size.png", dpi=250); plt.close()

    # ------------------------------------------------------------------
    # 7. BER and rate on the true channel using estimated-CSI beams.
    # ------------------------------------------------------------------
    print("Evaluating BER and achieved spectral efficiency...")
    ber_result, rate_result = evaluate_ber_and_rate_vs_snr(cfg, run, scale, models_by_seed)
    checkpoint("07_ber_rate", {"ber": ber_result, "rate": rate_result})
    plt.figure(figsize=(9, 6))
    for m, data in ber_result.items():
        vals = np.asarray(data["mean"], dtype=float)
        plt.semilogy(run.snr_points_db, np.maximum(vals, 1e-8), marker="o", label=m)
    plt.xlabel("Data/pilot SNR (dB)"); plt.ylabel("BER")
    plt.title("QPSK BER using estimated-CSI beams over the true channel")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out_dir / "06_ber_vs_snr.png", dpi=250); plt.close()
    plt.figure(figsize=(9, 6))
    for m, data in rate_result.items():
        vals = np.asarray(data["mean"], dtype=float)
        ci = np.asarray(data.get("ci95", np.zeros_like(vals)), dtype=float)
        plt.plot(run.snr_points_db, vals, marker="o", label=m)
        if np.any(ci > 0):
            plt.fill_between(run.snr_points_db, vals-ci, vals+ci, alpha=0.10)
    plt.xlabel("Data/pilot SNR (dB)"); plt.ylabel("Achieved spectral efficiency (bit/s/Hz)")
    plt.title("Achieved rate on the true channel using estimated CSI")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out_dir / "07_se_vs_snr.png", dpi=250); plt.close()

    # ------------------------------------------------------------------
    # 7b. BER and rate at the SAME nonzero aging horizon used above, so the
    #     predictive NMSE advantage is connected to a system-level payoff
    #     rather than only reported as an NMSE-only claim.
    # ------------------------------------------------------------------
    print(f"Evaluating BER and achieved spectral efficiency at {run.headline_prediction_delay_ms} ms aging...")
    ber_result_delayed, rate_result_delayed = evaluate_ber_and_rate_vs_snr(
        cfg, run, scale, models_by_seed, delay_ms=run.headline_prediction_delay_ms
    )
    checkpoint("07b_ber_rate_delayed", {"ber": ber_result_delayed, "rate": rate_result_delayed})
    plt.figure(figsize=(9, 6))
    for m, data in ber_result_delayed.items():
        vals = np.asarray(data["mean"], dtype=float)
        plt.semilogy(run.snr_points_db, np.maximum(vals, 1e-8), marker="o", label=m)
    plt.xlabel("Data/pilot SNR (dB)"); plt.ylabel("BER")
    plt.title(f"QPSK BER using estimated-CSI beams at {run.headline_prediction_delay_ms} ms aging")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out_dir / "06b_ber_vs_snr_delayed.png", dpi=250); plt.close()
    plt.figure(figsize=(9, 6))
    for m, data in rate_result_delayed.items():
        vals = np.asarray(data["mean"], dtype=float)
        ci = np.asarray(data.get("ci95", np.zeros_like(vals)), dtype=float)
        plt.plot(run.snr_points_db, vals, marker="o", label=m)
        if np.any(ci > 0):
            plt.fill_between(run.snr_points_db, vals-ci, vals+ci, alpha=0.10)
    plt.xlabel("Data/pilot SNR (dB)"); plt.ylabel("Achieved spectral efficiency (bit/s/Hz)")
    plt.title(f"Achieved rate using estimated CSI at {run.headline_prediction_delay_ms} ms aging")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out_dir / "07b_se_vs_snr_delayed.png", dpi=250); plt.close()

    # ------------------------------------------------------------------
    # 8. Complexity.
    # ------------------------------------------------------------------
    print("Measuring model complexity...")
    rng_c = np.random.default_rng(90_000)
    x_sp, _, _, _ = build_spatial_dataset(cfg, run, 16, rng_c, scale, fixed_snr_db=15.0, fixed_pilot_length=run.main_pilot_length)
    x_t, _, _, _ = build_temporal_dataset(cfg, run, 16, rng_c, scale, fixed_snr_db=15.0, fixed_pilot_length=run.main_pilot_length)
    complexity = {
        "CDRN-inspired baseline": measure_complexity(primary["cdrn"], x_sp),
        "Prop-Spatial": measure_complexity(primary["spatial"], x_sp),
        "CNN-BiLSTM-inspired baseline": measure_complexity(primary["cnn_bilstm"], x_t),
        "Proposed physics-guided temporal": measure_complexity(primary["proposed_prob"], x_t),
    }
    if run.include_gan_baseline: complexity["MultiScaleCNN GAN-inspired baseline"] = measure_complexity(primary["multiscale"], x_sp)
    if run.include_ode_baseline: complexity["GRU-ODE-inspired temporal baseline"] = measure_complexity(primary["ode_rnn"], x_t)
    pd.DataFrame(complexity).T.to_csv(out_dir / "08_complexity.csv")

    # ------------------------------------------------------------------
    # 9. Ablation with automatic substantive-effect flags.
    # ------------------------------------------------------------------
    print("Running physics-guided temporal ablation study...")
    ablation = run_temporal_ablation(cfg, run, scale, primary["spatial"])
    names = list(ablation.keys()); vals = [ablation[n]["mean"] for n in names]; cis = [ablation[n]["ci95"] for n in names]
    plt.figure(figsize=(12, 6)); plt.bar(np.arange(len(names)), vals, yerr=cis, capsize=3)
    plt.xticks(np.arange(len(names)), names, rotation=25, ha="right"); plt.ylabel("NMSE (dB)")
    plt.title("Physics-guided predictor ablation at 400 km/h, 10 dB, 1 ms")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(out_dir / "08_temporal_ablation.png", dpi=250); plt.close()

    # ------------------------------------------------------------------
    # 10. Robustness: K, phase errors, pathloss, and IRS strategies.
    # ------------------------------------------------------------------
    print("Evaluating model mismatch and IRS hardware robustness...")
    k_result = evaluate_k_mismatch(cfg, run, scale, primary)
    checkpoint("09_k_mismatch", k_result)
    plt.figure(figsize=(9, 6))
    for m, vals in k_result.items(): plt.plot(run.k_offsets_db, vals, marker="o", label=m)
    plt.xlabel("Rician K-factor offset from nominal training (dB)"); plt.ylabel("NMSE (dB)")
    plt.title("Generalization under Rician-K mismatch"); plt.grid(True, alpha=0.3); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_dir / "09_k_factor_mismatch.png", dpi=250); plt.close()

    phase_result = evaluate_phase_error_robustness(cfg, run, scale, primary)
    checkpoint("10_phase_error", phase_result)
    plt.figure(figsize=(9, 6))
    for m, vals in phase_result.items(): plt.plot(run.phase_error_std_deg, vals, marker="o", label=m)
    plt.xlabel("IRS phase-error standard deviation (degrees)"); plt.ylabel("NMSE (dB)")
    plt.title("Robustness to IRS phase errors"); plt.grid(True, alpha=0.3); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_dir / "10_phase_error_robustness.png", dpi=250); plt.close()

    pathloss_result = evaluate_pathloss_mismatch(cfg, run, scale, primary)
    checkpoint("pathloss_mismatch", pathloss_result)
    irs_phase_strategy = evaluate_irs_phase_strategy(cfg, run)

    # ------------------------------------------------------------------
    # 11. Wideband MIMO-OFDM extension (shared per-subcarrier weights).
    # ------------------------------------------------------------------
    ofdm_result = None
    if run.include_ofdm_extension:
        print("Evaluating wideband MIMO-OFDM extension...")
        ofdm_result = evaluate_ofdm_wideband_extension(cfg, run, scale, primary)
        with open(out_dir / "13_ofdm_wideband_extension.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(ofdm_result), f, indent=2)

    # ------------------------------------------------------------------
    # 12. Cross-scenario domain generalization.
    # ------------------------------------------------------------------
    print("Evaluating open-track <-> tunnel/cutting cross-scenario generalization...")
    cross_scenario = evaluate_cross_scenario_generalization(run)

    # ------------------------------------------------------------------
    # Numeric exports + consolidated readiness table.
    # ------------------------------------------------------------------
    all_results = {
        "nmse_vs_snr": nmse_snr,
        "nmse_vs_snr_delayed": nmse_snr_delayed,
        "headline_prediction_delay_ms": run.headline_prediction_delay_ms,
        "nmse_vs_speed": speed_result,
        "speed_points_kmh": list(run.speed_points_kmh),
        "channel_aging": aging_result,
        "aging_delays_ms": list(run.aging_delays_ms),
        "pilot_efficiency": pilot_result,
        "closed_loop_chain": closed_loop,
        "irs_scalability": irs_result,
        "ber_vs_snr": ber_result,
        "rate_vs_snr": rate_result,
        "ber_vs_snr_delayed": ber_result_delayed,
        "rate_vs_snr_delayed": rate_result_delayed,
        "complexity": complexity,
        "temporal_ablation": ablation,
        "k_factor_mismatch": k_result,
        "phase_error_robustness": phase_result,
        "pathloss_mismatch": pathloss_result,
        "irs_phase_strategy": irs_phase_strategy,
        "ofdm_wideband_extension": ofdm_result,
        "cross_scenario_generalization": cross_scenario,
        "global_channel_scale": scale,
    }
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(all_results), f, indent=2)

    summary_df = build_robustness_summary_table(all_results)
    summary_df.to_csv(out_dir / "14_robustness_summary_table.csv", index=False)
    with open(out_dir / "14_robustness_summary_table.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary_df.to_dict(orient="records")), f, indent=2)
    save_robustness_summary_figure(summary_df, out_dir / "14_robustness_summary_heatmap.png")

    save_dict_table(out_dir / "01_nmse_vs_snr_means.csv", "SNR_dB", run.snr_points_db, {m: d["mean"] for m, d in nmse_snr.items()})
    save_dict_table(out_dir / "02_nmse_vs_speed_means.csv", "speed_kmh", run.speed_points_kmh, {m: d["mean"] for m, d in speed_result.items()})
    save_dict_table(out_dir / "03_channel_aging_means.csv", "delay_ms", run.aging_delays_ms, {m: d["mean"] for m, d in aging_result.items()})
    save_dict_table(out_dir / "06_ber_vs_snr.csv", "SNR_dB", run.snr_points_db, {m: d["mean"] for m, d in ber_result.items()})
    save_dict_table(out_dir / "07_rate_vs_snr.csv", "SNR_dB", run.snr_points_db, {m: d["mean"] for m, d in rate_result.items()})
    save_dict_table(out_dir / "01c_nmse_vs_snr_delayed_means.csv", "SNR_dB", run.snr_points_db, {m: d["mean"] for m, d in nmse_snr_delayed.items()})
    save_dict_table(out_dir / "06b_ber_vs_snr_delayed.csv", "SNR_dB", run.snr_points_db, {m: d["mean"] for m, d in ber_result_delayed.items()})
    save_dict_table(out_dir / "07b_rate_vs_snr_delayed.csv", "SNR_dB", run.snr_points_db, {m: d["mean"] for m, d in rate_result_delayed.items()})

    print(f"\nCompleted. All figures, numeric results, calibration, closed-loop evaluations, and metadata saved to: {out_dir.resolve()}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rohitash Transactions-ready HSR IRS publication pipeline v6")
    p.add_argument(
        "--mode",
        choices=("publication", "smoke"),
        default="publication",
        help="publication = full research run; smoke = fast implementation validation",
    )
    p.add_argument("--output-dir", default=None, help="optional result directory override")
    p.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="override RunConfig.n_seeds (main-model seed count); publication default is 5",
    )
    p.add_argument(
        "--prediction-delay-ms",
        type=float,
        default=None,
        help="override the headline nonzero prediction horizon used for the "
        "delayed NMSE-vs-SNR and BER/SE-vs-SNR figures (default 1.0 ms)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SysConfig()
    run = RunConfig() if args.mode == "publication" else RunConfig.smoke()
    if args.output_dir:
        run.output_dir = args.output_dir
    if args.seeds is not None:
        run.n_seeds = args.seeds
    if args.prediction_delay_ms is not None:
        run.headline_prediction_delay_ms = args.prediction_delay_ms
    if args.mode == "publication" and run.n_seeds < 5:
        print(
            f"[WARNING] n_seeds={run.n_seeds} < 5 for a publication run. "
            "Statistical claims (95% CI, significance flags) in the reviewer-"
            "facing readiness table assume >=5 seeds; results from a smaller "
            "seed count should not be reported as final Transactions-grade evidence."
        )
    run_pipeline(cfg, run)


if __name__ == "__main__":
    main()