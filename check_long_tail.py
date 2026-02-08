import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def detect_long_tail(csv_path,
                     tail_ratio=100,
                     skew_thr=3,
                     kurt_thr=10,
                     need_two_rules=True):
    """
    自动检测长尾列
    -----------------------------------
    Parameters
    ----------
    csv_path : str
        CSV 文件路径（首列 id / 非数字列会自动忽略）
    tail_ratio : float
        max / median ≥ tail_ratio 视为“极值悬殊”
    skew_thr : float
        |skew| ≥ skew_thr 判作偏度过大
    kurt_thr : float
        kurtosis ≥ kurt_thr 判作峰度过大
    need_two_rules : bool
        是否要求至少满足“两条规则”才算长尾；False=只要满足一条
    -----------------------------------
    Returns
    -------
    long_tail_cols : list[str]
        检测出的长尾列名
    """
    df = pd.read_csv(csv_path, index_col=0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    long_tail_cols = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        max_med = series.max() / (series.median() + 1e-9)
        sk = skew(series)
        ku = kurtosis(series, fisher=False)

        rules = [
            max_med >= tail_ratio,
            abs(sk) >= skew_thr,
            ku >= kurt_thr
        ]
        if (sum(rules) >= 2 if need_two_rules else any(rules)):
            long_tail_cols.append(col)
            print(f"{col:<15} | max/med={max_med:>9.2g} | "
                  f"skew={sk:>8.2f} | kurt={ku:>8.2f}  <-- 长尾")
        else:
            print(f"{col:<15} | max/med={max_med:>9.2g} | "
                  f"skew={sk:>8.2f} | kurt={ku:>8.2f}")

    print("\n🔎 检测完毕，长尾列：", long_tail_cols)
    return long_tail_cols


# 调用示例
if __name__ == "__main__":
    long_cols = detect_long_tail("/root/autodl-tmp/QPP-Detection/myqpp1/qpp_features.csv")
    # -> 在脚本里直接用 LONG_TAIL_COLS = long_cols
