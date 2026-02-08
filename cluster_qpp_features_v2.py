import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_features(csv_path: Path):
    df = pd.read_csv(csv_path, index_col="id")
    if df.isna().all().any():
        print("⚠️  注意：存在整列全 NaN，请检查特征提取脚本")
    return df

LONG_TAIL_COLS = ['P_emd', 'P_fit', 'tau_fit', 'Q_fit', 'A0_fit', 'tau_env', 'kurt', 'dt_median']
FLAG_COL = "tau_fail"

def preprocess(df: pd.DataFrame):
    df = df.copy()
    # τ_env<0 flag
    if "tau_env" in df.columns:
        df[FLAG_COL] = (df["tau_env"] < 0).astype(int)
    # log10(abs)+1 on long-tail cols
    for col in LONG_TAIL_COLS:
        if col in df.columns:
            df[col] = np.log10(df[col].abs() + 1)
    # impute + robust scale
    imp = SimpleImputer(strategy="median")
    scl = RobustScaler(with_centering=True, with_scaling=True)
    X_imp = imp.fit_transform(df)
    X_scaled = scl.fit_transform(X_imp)
    return X_scaled, df.columns.tolist(), imp, scl


def reduce_pca(X, n_components=10, random_state=0):
    """PCA to n_components dimensions"""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    print(f"→ PCA 降到 {n_components} 维，累计方差贡献 {pca.explained_variance_ratio_.sum():.3f}")
    return X_pca, pca


def run_hdbscan(X, min_cluster_size=30, min_samples=10):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X)
    # silhouette only on core points
    mask = labels >= 0
    if mask.sum() > 1:
        sil = silhouette_score(X[mask], labels[mask])
    else:
        sil = np.nan
    print(f"[HDBSCAN] 簇数={len(set(labels)) - ( -1 in labels )}  Silhouette={sil:.3f}")
    return labels, sil


def run_spectral(X, n_clusters, n_neighbors=15, random_state=0):
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    labels = sc.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
    print(f"[Spectral k={n_clusters}] Silhouette={sil:.3f}")
    return labels, sil


def plot_umap(X_vis, labels, outfile="cluster_umap.png"):
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(uniq))
    plt.figure(figsize=(6,5))
    for i, lab in enumerate(uniq):
        m = labels == lab
        plt.scatter(X_vis[m,0], X_vis[m,1],
                    s=6, alpha=0.7,
                    color=cmap(i),
                    label=f"Cluster {lab}" if lab != -1 else "Noise")
    plt.axis("off")
    plt.legend(markerscale=3, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"📈 UMAP 散点图已保存：{outfile}")


def main():
    p = argparse.ArgumentParser(
        description="Cluster QPP features with HDBSCAN & Spectral Clustering"
    )
    p.add_argument("csv_file", help="输入 qpp_features.csv")
    p.add_argument("--out-csv", default="qpp_features_clustered.csv",
                   help="输出含标签的 CSV 文件名")
    args = p.parse_args()

    # load & preprocess
    df = load_features(Path(args.csv_file))
    X, feature_names, imp, scl = preprocess(df)

    # PCA to 10 dims
    X_pca, pca = reduce_pca(X, n_components=10)

    # grid search
    best = ("none", None, -1.0)
    # HDBSCAN
    lbl_h, sil_h = run_hdbscan(X_pca, min_cluster_size=30, min_samples=10)
    if sil_h > best[2]:
        best = ("hdbscan", lbl_h, sil_h)
    # Spectral k=3..5
    for k in [3,4,5]:
        lbl_s, sil_s = run_spectral(X_pca, n_clusters=k, n_neighbors=15)
        if sil_s > best[2]:
            best = (f"spectral_{k}", lbl_s, sil_s)

    method, labels, sil = best
    print(f"\n>>> 最优方法: {method}, silhouette={sil:.3f}")

    # UMAP 2D for viz
    reducer2 = umap.UMAP(n_components=2, random_state=0)
    X_vis = reducer2.fit_transform(X_pca)
    plot_umap(X_vis, labels)

    # save
    df["cluster"] = labels
    df.to_csv(args.out_csv)
    print(f"✅ 聚类标签 ({method}) 已写入 {args.out_csv}")

    # print summary
    print("\n各簇主要特征均值：")
    cols = ["P_cwt","Q_fit","tau_env","n_pulse"]
    print(df[df.cluster>=0].groupby("cluster")[cols].mean().round(3))


if __name__ == "__main__":
    main()

"""
python cluster_qpp_features_v2.py /root/autodl-tmp/QPP-Detection/myqpp1/csv/qpp_features.csv --out-csv qpp_clustered_v2.csv
"""