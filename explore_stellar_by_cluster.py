#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_stellar_by_cluster.py
探索不同 QPP 簇对应的恒星宏观属性（Prot, Teff, Mass, logg）差异
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("/root/autodl-tmp/QPP-Detection/myqpp1/1.csv", index_col="id")
    
    cluster_col = "cluster"
    
    props = ["Prot", "Teff", "Mass", "logg"]
    summary = df.groupby(cluster_col)[props].describe().round(3)
    print("\n=== 各簇恒星宏观属性描述 ===")
    print(summary)

    sns.set(style="whitegrid", font_scale=1.0)
    
    for prop in props:
        plt.figure(figsize=(6,4))
        plt.subplot(1,2,1)
        sns.boxplot(x=cluster_col, y=prop, data=df, palette="Set2")
        plt.title(f"{prop} 箱线图")
        plt.xlabel("Cluster")
        plt.ylabel(prop)
        
        plt.subplot(1,2,2)
        sns.violinplot(x=cluster_col, y=prop, data=df, palette="Set2", inner="quartile")
        plt.title(f"{prop} 小提琴图")
        plt.xlabel("Cluster")
        plt.ylabel(prop)
        
        plt.tight_layout()
        out_png = f"{prop}_by_cluster.png"
        plt.savefig(out_png, dpi=300)
        print(f"📈 已保存：{out_png}")
        plt.close()

if __name__ == "__main__":
    main()
