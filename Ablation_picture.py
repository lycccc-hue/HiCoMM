import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术图表样式
plt.style.use('seaborn-whitegrid')
sns.set_palette("deep")
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.titlesize': 12,
    'axes.labelsize': 10
})

# 表格数据
weibo_data = {
    'Method': ['w/o B', 'w/o V', 'w/o T', 'w/o C', 'w/Swin-T', 'w/o CO', 'w/o D', 'Full'],
    'Accuracy': [0.818, 0.911, 0.903, 0.929, 0.927, 0.937, 0.881, 0.942],
    'Precision': [0.793, 0.958, 0.956, 0.952, 0.947, 0.944, 0.970, 0.962],
    'Recall': [0.839, 0.859, 0.874, 0.914, 0.902, 0.868, 0.804, 0.926],
    'F1': [0.815, 0.906, 0.914, 0.933, 0.924, 0.904, 0.879, 0.943]
}

twitter_data = {
    'Method': ['w/o B', 'w/o V', 'w/o T', 'w/o C', 'w/Swin-T', 'w/o CO', 'w/o D', 'Full'],
    'Accuracy': [0.918, 0.882, 0.889, 0.915, 0.949, 0.866, 0.905, 0.966],
    'Precision': [0.898, 0.793, 0.911, 0.898, 0.965, 0.891, 0.884, 0.970],
    'Recall': [0.962, 0.932, 0.968, 0.965, 0.938, 0.945, 0.934, 0.961],
    'F1': [0.929, 0.857, 0.933, 0.932, 0.951, 0.917, 0.908, 0.965]
}

# 颜色设置 (保持原图配色)
colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c']  # 蓝/红/紫/绿
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
width = 0.18
x = np.arange(len(weibo_data['Method']))

# ===================== 微博数据集单独图片 =====================
plt.figure(figsize=(12, 6), dpi=120)
ax1 = plt.gca()

for i, metric in enumerate(metrics):
    ax1.bar(x + i * width, weibo_data[metric], width,
            label=metric, color=colors[i], edgecolor='white', linewidth=0.5)

    # 标记最大值（加粗显示）
    max_val = max(weibo_data[metric])
    if max_val in weibo_data[metric]:
        idx = weibo_data[metric].index(max_val)
        ax1.bar(x[idx] + i * width, max_val, width,
                color=colors[i], edgecolor='black', linewidth=1.5)

# plt.title('Ablation Study on Weibo Dataset', pad=12, fontweight='bold')
plt.xticks(x + width * 1.5, weibo_data['Method'], rotation=20, ha='right')
plt.ylabel('Score', labelpad=10)
plt.ylim(0.75, 1.0)
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax1.grid(axis='y', linestyle=':', alpha=0.7)

# 图例优化
plt.legend(ncol=4, bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=True)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# 保存高清图
plt.savefig('ablation_study_weibo.png', bbox_inches='tight', dpi=300)
plt.show()

# ===================== 推特数据集单独图片 =====================
plt.figure(figsize=(12, 6), dpi=120)
ax2 = plt.gca()

for i, metric in enumerate(metrics):
    ax2.bar(x + i * width, twitter_data[metric], width,
            color=colors[i], label=metric, edgecolor='white', linewidth=0.5)

    # 标记最大值
    max_val = max(twitter_data[metric])
    if max_val in twitter_data[metric]:
        idx = twitter_data[metric].index(max_val)
        ax2.bar(x[idx] + i * width, max_val, width,
                color=colors[i], edgecolor='black', linewidth=1.5)

# plt.title('Ablation Study on Twitter Dataset', pad=12, fontweight='bold')
plt.xticks(x + width * 1.5, twitter_data['Method'], rotation=20, ha='right')
plt.ylabel('Score', labelpad=10)
plt.ylim(0.75, 1.0)
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax2.grid(axis='y', linestyle=':', alpha=0.7)

# 图例优化
plt.legend(ncol=4, bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=True)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# 保存高清图
plt.savefig('ablation_study_twitter.png', bbox_inches='tight', dpi=300)
plt.show()