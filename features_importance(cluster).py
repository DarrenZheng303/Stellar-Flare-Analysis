#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_importance_full.py

将 qpp_clustered_v2.csv 中的所有数值特征（除 cluster 外）
和 qpp_with_stellar_and_activity.csv 中的活动特征合并，
对比簇 0/1，输出特征重要性排序与两簇均值。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    path_cluster = "/root/autodl-tmp/QPP-Detection/myqpp1/csv/qpp_clustered_v2.csv"
    path_activity = "/root/autodl-tmp/QPP-Detection/myqpp1/csv/qpp_with_stellar_and_activity.csv"

    df_cluster = pd.read_csv(path_cluster, index_col="id")
    df_act     = pd.read_csv(path_activity, index_col="id")

    df = df_cluster.join(df_act, how="inner")

    df = df[df["cluster"].isin([0,1])].copy()

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric.remove("cluster")

    X = df[numeric].fillna(0)  
    y = df["cluster"].values

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=0,
        n_jobs=-1
    )
    clf.fit(X, y)

    importances = pd.Series(
        clf.feature_importances_,
        index=numeric
    ).sort_values(ascending=False)

    means = df.groupby("cluster")[numeric].mean()

    print("=== 特征重要性 (降序) 及两簇均值 ===")
    header = f"{'feature':<25} {'importance':>10}  {'mean_clu0':>12}  {'mean_clu1':>12}"
    print(header)
    print("-"*len(header))
    for feat, imp in importances.items():
        m0 = means.loc[0, feat]
        m1 = means.loc[1, feat]
        print(f"{feat:<25} {imp:10.4f}  {m0:12.4g}  {m1:12.4g}")

    df_out = pd.DataFrame({
        "importance": importances,
        "mean_clu0": means.loc[0, importances.index].values,
        "mean_clu1": means.loc[1, importances.index].values
    }, index=importances.index)
    df_out.to_csv("feature_importance_by_cluster_full.csv")
    print("\n✅ 已保存完整的特征重要性和均值到 feature_importance_by_cluster_full.csv")

if __name__ == "__main__":
    main()
