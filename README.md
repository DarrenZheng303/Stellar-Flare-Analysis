# QPP Feature Extraction and Analysis

A comprehensive toolkit for extracting, analyzing, and clustering Quasi-Periodic Pulsations (QPP) from stellar flare light curves. This project provides advanced signal processing methods to characterize oscillatory patterns in stellar flare data from Kepler/K2 missions.

## Overview

This repository contains Python tools for:
- **Feature Extraction**: Extract multi-scale temporal and statistical features from stellar flare light curves using wavelet analysis, EMD, damped oscillation fitting, and nonlinear dynamics
- **Clustering Analysis**: Apply HDBSCAN and Spectral Clustering to identify distinct QPP patterns
- **Activity Analysis**: Compute stellar activity proxies including flare rates, spot modulation amplitudes, and stellar rotation parameters
- **Feature Importance**: Identify key features that distinguish different QPP clusters

## Features

### Multi-Scale Signal Processing
- **Continuous Wavelet Transform (CWT)**: Dominant period, power spectrum, peak width, and frequency drift analysis
- **Empirical Mode Decomposition (EMD)**: Instantaneous frequency and quality factor estimation
- **Damped Oscillation Fitting**: Extract decay time, period, and quality factor via nonlinear least squares
- **Hilbert Transform**: Envelope extraction and modulation index computation

### Nonlinear & Statistical Analysis
- Sample entropy and spectral entropy
- Lempel-Ziv complexity
- Hurst exponent (long-range dependence)
- Higher-order moments (skewness, kurtosis)
- Pulse detection and inter-pulse timing statistics

### Machine Learning
- **Dimensionality Reduction**: PCA and UMAP
- **Clustering**: HDBSCAN (density-based) and Spectral Clustering
- **Feature Selection**: Random Forest feature importance ranking

## Installation

### Prerequisites
- Python 3.8+
- FITS file reader (Astropy)
- Scientific computing libraries

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
astropy>=5.0
scikit-learn>=1.0
matplotlib>=3.4.0
pycwt>=0.3.0a22
PyEMD>=1.3.0
antropy>=0.1.4
nolds>=0.5.2
hdbscan>=0.8.27
umap-learn>=0.5.1
```

## Usage

### 1. Extract QPP Features

Extract features from FITS light curve files:

```bash
python extract_qpp_features.py /path/to/fits_files -o qpp_features.csv
```

**Options:**
- `fits_dir`: Directory containing `*.fits` files
- `-o, --outfile`: Output CSV filename (default: `qpp_features.csv`)
- `--time-col`: Time column name in FITS (default: `TIME`)
- `--flux-col`: Flux column name in FITS (default: `FLUX`)

**Output:** CSV file with extracted features including:
- `P_cwt`, `power_cwt`, `peak_w_cwt`, `freq_slope` (wavelet features)
- `P_emd`, `Q_emd` (EMD features)
- `P_fit`, `tau_fit`, `Q_fit`, `A0_fit` (damped oscillation fit)
- `tau_env`, `mod_index` (envelope features)
- `skew`, `kurt`, `samp_entropy`, `spec_entropy`, `lz_complex`, `hurst` (nonlinear features)
- `n_pulse`, `dt_median`, `dt_std` (pulse statistics)

### 2. Cluster QPP Features

Identify distinct QPP patterns using clustering:

```bash
python cluster_qpp_features_v2.py qpp_features.csv --out-csv qpp_clustered.csv
```

**Options:**
- `csv_file`: Input feature CSV from step 1
- `--out-csv`: Output CSV with cluster labels (default: `qpp_features_clustered.csv`)

**Output:**
- CSV file with additional `cluster` column
- UMAP visualization (`cluster_umap.png`)
- Silhouette scores and cluster statistics

### 3. Analyze Stellar Activity

Compute activity proxies and merge with stellar parameters:

```bash
python analyze_activity.py /path/to/fits_files --planet-data planet_data.txt -o activity_summary.csv
```

**Options:**
- `fits_dir`: Directory containing FITS files
- `--planet-data`: Stellar rotation and Ca II S-index table (space-separated)
- `-o, --out`: Output CSV filename (default: `qpp_with_stellar_and_activity.csv`)

**Output:** CSV with:
- Flare rate and energy distribution
- Spot modulation amplitude and filling factor
- Stellar parameters (Teff, log g, Prot, etc.)

### 4. Feature Importance Analysis

Identify which features best distinguish clusters:

```bash
python features_importance\(cluster\).py
```

**Note:** Edit file paths in the script to point to your clustered and activity CSV files.

**Output:**
- Console output showing ranked feature importance
- `feature_importance_by_cluster_full.csv` with feature rankings and cluster means

### 5. Additional Scripts

- **`check_long_tail.py`**: Visualize feature distributions and identify long-tailed variables
- **`cross_validate_activity_cluster.py.py`**: Cross-validation for activity-based clustering
- **`explore_stellar_by_cluster.py`**: Exploratory analysis of stellar parameters by cluster

## Data Format

### Input FITS Files
- Extension 1: Binary table with columns `TIME` and `FLUX`
- Example: Kepler/K2 light curves from MAST archive

### Stellar Parameter File (`planet_data.txt`)
Space-separated table with columns:
```
KIC Teff logg Mass Prot sigmaP Rper LPH w DC Flag
```

## Methodology

### Feature Extraction Pipeline
1. **Load & Clean**: Read FITS light curves, remove NaNs
2. **Detrend**: Apply median filter to remove long-term trends
3. **Normalize**: Scale by standard deviation
4. **Multi-Scale Analysis**:
   - CWT with Morlet wavelet to extract dominant periods
   - EMD to decompose into intrinsic mode functions
   - Fit damped cosine to quantify decay timescales
5. **Nonlinear Dynamics**: Compute entropy, complexity, and fractal dimension
6. **Pulse Statistics**: Detect peaks and measure inter-pulse intervals

### Clustering Strategy
1. **Preprocessing**:
   - Log-transform long-tailed features
   - Impute missing values (median)
   - Robust scaling (RobustScaler)
2. **Dimensionality Reduction**: PCA to 10 components
3. **Clustering**: Grid search over HDBSCAN and Spectral Clustering parameters
4. **Selection**: Choose method with highest silhouette score
5. **Visualization**: UMAP 2D projection for cluster visualization

## Project Structure

```
QPP-ShikangZheng/
├── extract_qpp_features.py          # Feature extraction from light curves
├── cluster_qpp_features_v2.py       # Clustering analysis
├── analyze_activity.py              # Activity proxy computation
├── features_importance(cluster).py  # Feature importance ranking
├── check_long_tail.py               # Distribution analysis
├── cross_validate_activity_cluster.py.py  # Cross-validation
├── explore_stellar_by_cluster.py    # Exploratory analysis
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Scientific Background

### Quasi-Periodic Pulsations (QPP)
QPPs are oscillatory patterns observed in stellar and solar flare light curves, with periods typically ranging from seconds to hours. These pulsations provide insights into:
- Magnetic field structure and dynamics
- Energy release and transport mechanisms
- Flare triggering and evolution

### Applications
- Characterizing flare physics across stellar populations
- Identifying connections between QPP properties and stellar activity
- Testing magnetohydrodynamic (MHD) oscillation models

## Citation

If you use this code in your research, please cite:

```
[Add your citation information here]
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries:
- GitHub Issues: https://github.com/DarrenZheng303/Stellar-Flare-Analysis/issues
- GitHub: [@DarrenZheng303](https://github.com/DarrenZheng303)

## Acknowledgments

This project uses data from:
- NASA Kepler/K2 missions
- MAST (Mikulski Archive for Space Telescopes)

## References

- Continuous Wavelet Transform: Torrence & Compo (1998)
- Empirical Mode Decomposition: Huang et al. (1998)
- HDBSCAN: McInnes et al. (2017)
- UMAP: McInnes et al. (2018)
