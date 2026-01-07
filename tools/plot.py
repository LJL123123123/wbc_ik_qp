import os
import matplotlib

# If no DISPLAY (headless), use non-interactive Agg backend so we can save files.
if os.getenv("DISPLAY") is None:
    matplotlib.use("Agg")
else:
    # prefer a GUI backend when a display is available
    matplotlib.use("TkAgg")

import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
parser.add_argument('arg', help='Path to the CSV file to plot.')
args = parser.parse_args()
arg = args.arg
# 读取CSV文件
csv_path = arg
df = pd.read_csv(csv_path)

# 假设第一列是'time'
time_col = df.columns[0]

# 指定要绘制的列名
# plot_columns = ['com_pos_target[0]', 'com_pos_target[1]', 'com_pos_target[2]', 'com_ori_target[0]', 'com_ori_target[1]', 'com_ori_target[2]']  # 替换为你要绘制的列名

# # 检查列名是否存在
# for col in plot_columns:
#     if col not in df.columns:
#         raise ValueError(f"列名 '{col}' 不在CSV文件中。")

# 绘图
plt.figure(figsize=(10, 6))
for col in df.columns:
    if col == time_col:
        continue
    plt.plot(df[time_col], df[col], label=col)

plt.xlabel(time_col)
plt.ylabel('Value')
plt.title('Plot of Selected Columns')
plt.legend()
plt.grid(True)
# Always save the figure to a file. If a display is available, also show it.
plt.tight_layout()
out = os.path.splitext(os.path.basename(csv_path))[0] + ".png"
plt.savefig(out, dpi=150)
print(f"Saved figure to {out}")

# Try to show interactively if a DISPLAY is available. Wrap in try/except
# so script still works in headless environments even if a GUI call fails.
try:
    if os.getenv("DISPLAY") is not None:
        plt.show()
except Exception as e:
    print(f"Could not show figure interactively: {e}")