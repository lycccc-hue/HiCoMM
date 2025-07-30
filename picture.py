import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 数据准备
combinations = ['3CNN+1Trans', '2CNN+2Trans', '1CNN+3Trans']
f1_weibo = [0.938, 0.943, 0.934]
f1_twitter = [0.957, 0.965, 0.963]
avg_f1 = [(w + t) / 2 for w, t in zip(f1_weibo, f1_twitter)]

# 创建画布（调整宽高比）
plt.figure(figsize=(10, 6), dpi=100)
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1. 坐标轴优化
plt.ylim(0.930, 0.970)  # 收紧y轴范围
plt.yticks([i/1000 for i in range(935, 975, 5)])  # 0.005间隔刻度
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  # 统一小数位数
plt.grid(axis='y', linestyle='--', alpha=0.5)  # 水平网格线

# 2. 绘制折线（加粗+透明度）
line_width = 1.5
plt.plot(combinations, f1_weibo, label='Weibo F1', color='blue',
         linewidth=line_width, alpha=0.85, marker='o')
plt.plot(combinations, f1_twitter, label='Twitter F1', color='green',
         linewidth=line_width, alpha=0.85, marker='s')
plt.plot(combinations, avg_f1, label='Average F1', color='orange',
         linewidth=line_width, alpha=0.85, marker='^', linestyle='--')

# 3. 突出最佳点（改用星形标记）
best_point = plt.scatter('2CNN+2Trans', avg_f1[1], color='red', marker='*',
                         s=200, label='Best (2CNN+2Trans)',
                         edgecolors='black', linewidths=0.5)

# 4. 添加参考线和标注
# plt.axhline(y=0.955, color='gray', linestyle=':', linewidth=1)
# plt.text(2.1, 0.956, 'SOTA Baseline (0.955)', fontsize=9, color='gray')

# 5. 标签和标题优化
plt.xlabel('Layer Combinations', labelpad=10)
plt.ylabel('F1 Score', labelpad=10)

# 6. 图例和文字标注
legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
# legend = plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95), framealpha=0.9)
# plt.text(0.02, 0.96, 'Max Gap: +0.018 (1CNN+3Trans)', transform=ax.transAxes,
#          fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

# 7. 旋转x轴标签
plt.xticks(rotation=15, ha='right')

# 调整边距
plt.tight_layout()
plt.show()