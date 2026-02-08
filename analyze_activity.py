#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_activity.py
批量分析 QPP 光变曲线的局部活动代理，并与恒星自转 & CaII S-index 合并，
输出 qpp_with_stellar_and_activity.csv

特征：
  1. 闪焰率 & 能量分布
  2. 星斑调制振幅 & 填充因子
  3. Ca II H&K S‑指数 (从 planet_data.txt 读取)

不包含 X‑射线 L_X
"""
import re
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def extract_kic(lc_id):
    """从 'KIC_1025986_event_...' 提取 '1025986'"""
    m = re.search(r'KIC[_\-]?(\d+)', lc_id)
    return m.group(1) if m else None


def load_lightcurve(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        t = np.asarray(data['TIME'], float)
        f = np.asarray(data['FLUX'], float)
    mask = np.isfinite(t) & np.isfinite(f)
    return t[mask], f[mask]


def compute_flare_features(t, f, sigma_thresh=5.0, min_duration=0.01):
    """计算闪焰率与闪焰能量分布"""
    f0 = f - np.median(f)
    sigma = np.std(f0)
    thr = sigma_thresh * sigma
    peaks, _ = find_peaks(f0, height=thr)
    if len(peaks) == 0:
        return {'flare_rate': 0.0, 'flare_energy_mean': np.nan, 'flare_energy_std': np.nan}
    # 连续峰值聚合
    energies = []
    dt = np.median(np.diff(t))
    for p in peaks:
        energy = f0[p] * dt  # 简易能量近似
        energies.append(energy)
    rate = len(peaks) / (t.max() - t.min())
    return {
        'flare_rate': rate,
        'flare_energy_mean': float(np.mean(energies)),
        'flare_energy_std': float(np.std(energies))
    }


def compute_spot_features(t, f, window_fraction=0.1):
    """计算星斑调制振幅与填充因子"""
    dt = np.median(np.diff(t))
    window = max(int(window_fraction * len(t)), 1)
    smooth = uniform_filter1d(f, size=window)
    amp = np.percentile(smooth, 95) - np.percentile(smooth, 5)
    fill = amp / np.median(smooth) if np.median(smooth) != 0 else np.nan
    return {'spot_amp': float(amp), 'spot_fill': float(fill)}


def main():
    parser = argparse.ArgumentParser(description='Analyze QPP activity proxies (no X-ray)')
    parser.add_argument('fits_dir', help='目录，包含 .fits 文件')
    parser.add_argument('--planet-data', default='planet_data.txt',
                        help='恒星自转 & CaII S-index 表，空格分隔')
    parser.add_argument('-o', '--out', default='qpp_with_stellar_and_activity.csv',
                        help='输出 CSV 文件')
    args = parser.parse_args()

    # 1. 计算活动代理
    rows = []
    for fpath in sorted(Path(args.fits_dir).glob('*.fits')):
        lc_id = fpath.stem
        t, f = load_lightcurve(fpath)
        if len(t) < 50:
            continue
        flare = compute_flare_features(t, f)
        spot = compute_spot_features(t, f)
        rows.append({'id': lc_id, **flare, **spot})
        print(f'✅ {lc_id}: flare_rate={flare["flare_rate"]:.3g}, spot_amp={spot["spot_amp"]:.3g}')
    df_act = pd.DataFrame(rows).set_index('id')

    # 2) 读取 planet_data.txt
    star_file = Path(args.planet_data)
    if not star_file.exists():
        raise FileNotFoundError(f"{star_file} 未找到")

    cols = ["KIC","Teff","logg","Mass","Prot","sigmaP","Rper","LPH","w","DC","Flag","Extra"]
    df_star = pd.read_csv(
        star_file, sep=r"\s+", header=None, names=cols, comment="#"
    ).drop(columns=["Extra"])
    df_star["KIC"] = df_star["KIC"].astype(str)

    # 3) 合并
    df = df_act.reset_index()
    df["KIC"] = df["id"].apply(extract_kic)
    df_merged = df.merge(df_star, on="KIC", how="left")

    # 4) 保存
    df_merged.to_csv(args.out, index=False, float_format="%.6g")
    print(f"🎉 输出已保存到 {args.out}")

if __name__ == '__main__':
    main()

"""
python analyze_activity.py /root/autodl-tmp/QPP-Detection/QPP_real_data_origin \
  --planet-data /root/autodl-tmp/QPP-Detection/myqpp1/planet_data.txt \
  -o qpp_with_stellar_and_activity.csv
"""