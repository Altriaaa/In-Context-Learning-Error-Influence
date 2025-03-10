import pandas as pd
import matplotlib.pyplot as plt

# 读取数据并重新排序
df = pd.read_csv("results\\final_result.csv")
control_row = df[df["config"] == "Control"]
other_rows = df[df["config"] != "Control"]
df_sorted = pd.concat([control_row, other_rows])

# 创建画布和双坐标轴
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# 绘制Accuracy（左轴）
ax1.plot(
    df_sorted["error_ratio"],
    df_sorted["accuracy"],
    marker="o",
    color="tab:blue",
    linewidth=2,
    markersize=8,
    label="Accuracy"
)

# 绘制EFR（右轴）
ax2.plot(
    df_sorted["error_ratio"],
    df_sorted["error_follow_ratio"],
    marker="s",
    color="tab:red", 
    linewidth=2,
    markersize=8,
    label="EFR"
)

# 设置坐标轴标签和样式
ax1.set_xlabel("Error Ratio", fontsize=12)
ax1.set_ylabel("Accuracy", color="tab:blue", fontsize=12)
ax2.set_ylabel("Error Follow Ratio", color="tab:red", fontsize=12)

# 设置x轴刻度为百分比格式
ax1.set_xticks(df_sorted["error_ratio"])
ax1.set_xticklabels([f"{x*100:.0f}%" for x in df_sorted["error_ratio"]], fontsize=10)

# 设置网格线和颜色透明度
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.title("Model Performance vs Error Ratio", pad=20, fontsize=14)
plt.tight_layout()
plt.show()