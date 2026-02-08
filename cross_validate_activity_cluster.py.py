#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_validate_activity_cluster.py

将 qpp_with_stellar_and_activity.csv 与 qpp_clustered.csv 合并，
并按簇统计每个活动特征（闪焰 & 星斑）的样本数、均值、标准差。
"""
import pandas as pd
import seaborn as sns

def main():

    df_act = pd.read_csv(
        "qpp_with_stellar_and_activity.csv",
        index_col="id",
        na_values=["", " ", "nan", "---"]
    )

    df_clu = pd.read_csv(
        "/root/autodl-tmp/QPP-Detection/myqpp1/csv/qpp_clustered_v2.csv",
        index_col="id"
    )

    df = df_act.join(df_clu["final_lbl"].rename("cluster"), how="inner")

    activity_cols = [
        "flare_rate", "flare_energy_mean", "flare_energy_std",
        "spot_amp", "spot_fill"
    ]

    stats = df.groupby("cluster")[activity_cols].agg(['count', 'mean', 'std'])

    pd.set_option("display.precision", 3)
    print("\n=== 各簇 Activity 特征 描述统计 ===\n")
    print(stats)

    stats.to_csv("activity_stats_by_cluster.csv")
    print("\n✅ 统计结果已保存到 activity_stats_by_cluster.csv")
    
    

if __name__ == "__main__":
    main()
