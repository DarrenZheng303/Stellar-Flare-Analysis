#!/usr/bin/env python3
# ------------------------------------------------------------
#  extract_qpp_features.py 
#  批量提取恒星 QPP 光变曲线的多尺度 / 统计特征
# ------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import signal, optimize, stats
import pycwt as cwt
from PyEMD import EMD
from scipy.signal import hilbert, peak_widths

import antropy as ant
import nolds


def load_lightcurve(path: Path, col_time="TIME", col_flux="FLUX"):
    with fits.open(path) as hdul:
        data = hdul[1].data
        time = np.asarray(data[col_time], dtype=float)
        flux = np.asarray(data[col_flux], dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]


def detrend_and_normalize(flux):
    flux = flux - signal.medfilt(flux, kernel_size=31)
    flux_std = np.std(flux) or 1.0
    return flux / flux_std


def wavelet_features(t, y):
    dt = np.median(np.diff(t))
    mother = cwt.Morlet(6)
    coef, scales, freqs, coi, *rest = cwt.cwt(
        y, dt, dj=1/12, s0=2*dt, J=-1, wavelet=mother
    )
    power = np.abs(coef) ** 2

    gws = power.mean(axis=1)
    idx_peak = np.argmax(gws)
    period = 1.0 / freqs[idx_peak]

    results_half = peak_widths(gws, [idx_peak], rel_height=0.5)
    width_idx = results_half[0][0]

    peak_w = width_idx * (period / idx_peak)  # 近似换算

    ridge = np.argmax(power, axis=0)          # 每一列最大功率尺度索引
    ridge_freq = freqs[ridge]

    mask = (t >= coi[ridge])                  # coi 同尺度
    if mask.sum() > 10:
        p = np.polyfit(t[mask], ridge_freq[mask], 1)   # 线性回归 freq(t)
        freq_slope = p[0]                      # Hz / s
    else:
        freq_slope = np.nan

    return {
        "P_cwt": period,
        "power_cwt": float(gws[idx_peak]),
        "peak_w_cwt": peak_w,
        "freq_slope": freq_slope
    }

def emd_features(t, y, fs):
    emd = EMD()
    imfs = emd(y)
    if imfs.size == 0:
        return {"P_emd": np.nan, "Q_emd": np.nan}

    energies = np.sum(imfs ** 2, axis=1)
    imf = imfs[np.argmax(energies)]

    analytic = hilbert(imf)
    inst_amp = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * fs / (2 * np.pi)
    if len(inst_freq) == 0 or np.median(inst_freq) <= 0:
        return {"P_emd": np.nan, "Q_emd": np.nan}

    P = 1.0 / np.median(inst_freq)
    try:
        idx_tau = np.where(inst_amp < inst_amp[0] / np.e)[0][0]
        tau = idx_tau / fs
        Q = np.pi * tau / P
    except IndexError:
        tau, Q = np.nan, np.nan
    return {"P_emd": P, "Q_emd": Q}


def _damped_cos(t, A0, tau, P, phi, C):
    return A0 * np.exp(-t / tau) * np.cos(2 * np.pi * t / P + phi) + C


def fit_damped(t, y, P0):
    guess = [np.ptp(y)/2, len(t)*np.median(np.diff(t))/4, P0, 0, 0]
    try:
        popt, _ = optimize.curve_fit(
            _damped_cos, t, y, p0=guess, maxfev=4000, bounds=(0, np.inf)
        )
        A0, tau, P, phi, C = popt
        Q = np.pi * tau / P
        return {"P_fit": P, "tau_fit": tau, "Q_fit": Q, "A0_fit": A0}
    except (RuntimeError, ValueError):
        return {"P_fit": np.nan, "tau_fit": np.nan,
                "Q_fit": np.nan, "A0_fit": np.nan}

def envelope_features(t, y):
    analytic = hilbert(y)
    env = np.abs(analytic)
    try:
        (A0, tau), _ = optimize.curve_fit(
            lambda x, A0, tau: A0*np.exp(-x/tau),
            t - t[0], env, p0=[env[0], (t[-1]-t[0])/4], maxfev=4000
        )
    except (RuntimeError, ValueError):
        tau = np.nan
    mod_index = (env.max() - env.min()) / (env.max() + env.min())
    return {"tau_env": tau, "mod_index": mod_index}


def nonlinear_features(y):
    y_norm = (y - np.mean(y)) / np.std(y)
    return {
        "skew": stats.skew(y_norm),
        "kurt": stats.kurtosis(y_norm, fisher=False),
        "samp_entropy": ant.sample_entropy(y_norm),
        "spec_entropy": ant.spectral_entropy(y_norm, sf=1, method='welch'),
        "lz_complex": ant.lziv_complexity((y_norm > 0).astype(int)),
        "hurst": nolds.hurst_rs(y_norm)
    }


def pulse_features(t, y, z=1.0):
    """阈值 = μ + z·σ；返回脉冲计数和 Δt 统计"""
    peaks, _ = signal.find_peaks(y, height=np.mean(y)+z*np.std(y))
    if len(peaks) >= 2:
        dt = np.diff(t[peaks])
        dt_median, dt_std = np.median(dt), np.std(dt)
    else:
        dt_median, dt_std = np.nan, np.nan
    return {
        "n_pulse": len(peaks),
        "dt_median": dt_median,
        "dt_std": dt_std
    }


def extract_features(lc_id, t, y):
    t = t - t[0]
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    y_clean = detrend_and_normalize(y)

    feats = {"id": lc_id}
    feats.update(wavelet_features(t, y_clean))
    feats.update(emd_features(t, y_clean, fs))
    feats.update(fit_damped(t, y_clean, feats["P_cwt"]))
    feats.update(envelope_features(t, y_clean))
    feats.update(nonlinear_features(y_clean))
    feats.update(pulse_features(t, y_clean))
    return feats


def main():
    parser = argparse.ArgumentParser(description="Batch extract QPP features.")
    parser.add_argument("fits_dir", type=str, help="Dir with *.fits files")
    parser.add_argument("-o", "--outfile", default="qpp_features.csv")
    parser.add_argument("--time-col", default="TIME")
    parser.add_argument("--flux-col", default="FLUX")
    args = parser.parse_args()

    fits_dir = Path(args.fits_dir)
    files = sorted(fits_dir.glob("*.fits"))
    if not files:
        print("❌  未找到任何 FITS 文件"); return

    rows = []
    for f in files:
        try:
            t, y = load_lightcurve(f, args.time_col, args.flux_col)
            if len(t) < 10: continue
            rows.append(extract_features(f.stem, t, y))
            print(f"✅ {f.name} done.")
        except Exception as e:
            print(f"⚠️  {f.name} failed: {e}")

    df = pd.DataFrame(rows).set_index("id")
    df.to_csv(args.outfile)
    print(f"\n🎉  Feature table saved to {args.outfile}")
    print(f"   Total curves processed: {len(rows)}")


if __name__ == "__main__":
    main()

"""
python /root/autodl-tmp/QPP-Detection/myqpp1/extract_qpp_features.py /root/autodl-tmp/QPP-Detection/QPP_real_data_origin -o qpp_features.csv
"""