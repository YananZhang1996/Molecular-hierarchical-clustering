import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import time
import os
from scipy.signal import find_peaks
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# -------------------------- 1. 数据读取与预处理 --------------------------
def read_data(file_path):
    """读取CSV文件并自动处理编码问题"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, index_col=0, encoding=encoding)
            print(f"成功读取文件，使用编码: {encoding}")
            print(f"数据形状：{df.shape}（样本数×特征数）")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用编码{encoding}读取失败: {str(e)}")

    raise ValueError("无法解析文件，请检查文件格式和编码")


# 读取数据
file_path = r'D:\AI_for_electrolyte\S\Input_feature2.csv'  # 替换为你的文件路径
df = read_data(file_path)

# 处理缺失值
if df.isnull().any().any():
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    print(f"检测到缺失值，占比: {missing_ratio:.2%}，使用特征均值填充")
    df = df.fillna(df.mean())

# 数据标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

# -------------------------- 2. 分层聚类计算 --------------------------
print("开始分层聚类计算...")
start_time = time.time()

# 计算距离矩阵（使用余弦距离适合高维数据）
distance_matrix = pdist(df_scaled, metric='cosine')

# 构建层次聚类树（使用ward方法）
linkage_matrix = linkage(distance_matrix, method='ward')

end_time = time.time()
print(f"分层聚类计算完成，耗时: {end_time - start_time:.2f}秒")

# -------------------------- 3. 自动确定聚类层次（保持自适应层数） --------------------------
print("根据数据特征自动确定聚类层次...")

# 提取合并高度
merge_heights = linkage_matrix[:, 2]
height_diff = np.diff(merge_heights)
peaks, _ = find_peaks(height_diff, height=np.percentile(height_diff, 75), distance=10)

# 保持自适应层数（8-12层）
if len(peaks) < 3:
    percentiles = np.linspace(95, 5, 5)
    additional_points = [np.percentile(merge_heights, p) for p in percentiles]
    additional_indices = [np.argmin(np.abs(merge_heights - p)) for p in additional_points]
    peaks = np.unique(np.concatenate([peaks, additional_indices]))

peaks = np.sort(peaks)
n_layers = min(max(len(peaks), 8), 12)  # 保持自适应层数不变
selected_peaks = peaks[-n_layers:]

# 计算所有自适应层次的聚类数（保持不变）
n_samples = df.shape[0]
cluster_counts = [n_samples - (peak + 1) for peak in selected_peaks]
cluster_counts = sorted(list(set(cluster_counts)), reverse=True)

# 确保至少有5层以便选择最后5层
if len(cluster_counts) < 5:
    cluster_counts = sorted([n_samples - int(n_samples * f) for f in np.linspace(0.1, 0.9, 8)], reverse=True)

print(f"自适应确定了 {len(cluster_counts)} 个聚类层次（保持不变）")
print(f"所有层次聚类数: {', '.join(map(str, cluster_counts))}")

# -------------------------- 4. 只提取倒数5层聚类结果 --------------------------
print("提取倒数5层聚类结果...")
start_time = time.time()

# 确保有至少5层，如果不足则调整
required_layers = 5
if len(cluster_counts) < required_layers:
    # 补充层数至至少5层
    additional = required_layers - len(cluster_counts)

    # 确保last_cluster始终有定义
    if cluster_counts:  # 如果cluster_counts不为空
        last_cluster = cluster_counts[-1]
    else:  # 如果cluster_counts为空，使用默认值
        last_cluster = max(2, n_samples // 2)  # 确保至少为2

    step = max(1, last_cluster // (additional + 1))
    additional_cluster_counts = []
    current = last_cluster - step

    while len(additional_cluster_counts) < additional and current >= 2:
        additional_cluster_counts.append(current)
        current -= step

    # 合并并去重，确保有足够的层数
    combined = list(set(cluster_counts + additional_cluster_counts))
    cluster_counts = sorted(combined, reverse=True)

# 选择倒数5层（最后5层）
selected_layers = cluster_counts[-5:]  # 核心修改：选择最后5层
total_layers = len(cluster_counts)
layer_indices = list(range(total_layers - 4, total_layers + 1))  # 层号：倒数第5到最后1层

results = []
for i, (layer_idx, n_clusters) in enumerate(zip(layer_indices, selected_layers)):
    labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    results.append(pd.DataFrame({
        '样本ID': df.index,
        f'第{layer_idx}层(聚类数={n_clusters})': labels
    }).set_index('样本ID'))

# 合并结果并保存
all_layers = pd.concat(results, axis=1)
output_dir = "自适应聚类结果_倒数5层_带热图"
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, "倒数5层聚类结果.xlsx")
all_layers.to_excel(excel_path, engine='openpyxl')

end_time = time.time()
print(f"倒数5层聚类结果已保存至: {excel_path}")

# -------------------------- 5. 绘制聚类树状图（标记倒数5层） --------------------------
print("绘制标记倒数5层的聚类树状图...")
plt.figure(figsize=(20, 12))

# 绘制树状图
dendro = dendrogram(
    linkage_matrix,
    orientation='top',
    truncate_mode='lastp',
    p=150,
    leaf_rotation=90,
    leaf_font_size=6,
    color_threshold=0.3 * max(linkage_matrix[:, 2]),
    above_threshold_color='lightgray'
)

# 只标记倒数5层
for i, (layer_idx, n_clusters) in enumerate(zip(layer_indices, selected_layers)):
    if n_clusters < n_samples:
        step = n_samples - n_clusters - 1
        if step >= 0 and step < linkage_matrix.shape[0]:
            height = linkage_matrix[step, 2]
            plt.axhline(y=height, color=f'C{i}', linestyle='--', alpha=0.7,
                        label=f'第{layer_idx}层 (k={n_clusters})')

# 美化树状图
plt.title('聚类树状图（标记倒数5层）', fontsize=18, pad=20)
plt.xlabel('样本（按聚类关系排序）', fontsize=14)
plt.ylabel('聚类距离（合并高度）', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存树状图
tree_path = os.path.join(output_dir, "倒数5层聚类树状图.png")
plt.tight_layout()
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"标记倒数5层的树状图已保存至: {tree_path}")

# -------------------------- 6. 添加聚类热图（使用最后一层聚类结果） --------------------------
print("绘制聚类热图...")

# 使用最后一层的聚类结果
last_layer_idx = layer_indices[-1]
last_layer_clusters = selected_layers[-1]
last_layer_labels = fcluster(linkage_matrix, t=last_layer_clusters, criterion='maxclust')

# 获取按聚类顺序排序的样本索引
sorted_leaves = leaves_list(linkage_matrix)
df_sorted = df_scaled.iloc[sorted_leaves]  # 按聚类结果排序数据

# 创建热图
plt.figure(figsize=(18, 12))
g = sns.heatmap(
    df_sorted,
    cmap='coolwarm',
    center=0,
    xticklabels=1,  # 每隔1个显示特征名
    yticklabels=False,  # 样本太多，不显示样本名
    cbar_kws={'label': '标准化特征值'}
)

# 添加聚类分隔线
unique_labels = sorted(np.unique(last_layer_labels))
label_indices = {label: [] for label in unique_labels}

# 构建标签位置映射
for i, leaf_idx in enumerate(sorted_leaves):
    label = last_layer_labels[leaf_idx]
    label_indices[label].append(i)

# 绘制类别分隔线
prev_max = -1
for label in unique_labels:
    if label_indices[label]:
        current_min = min(label_indices[label])
        if prev_max >= 0 and current_min > prev_max:
            plt.axhline(y=prev_max + 0.5, color='black', linewidth=2,
                        xmin=0, xmax=1)
        prev_max = max(label_indices[label])

plt.title(f'聚类热图（按最后一层（第{last_layer_idx}层）聚类结果排序）', fontsize=16, pad=20)
plt.xlabel('特征', fontsize=14)
plt.ylabel('样本（按聚类顺序排列）', fontsize=14)

# 保存热图
heatmap_path = os.path.join(output_dir, f'最后一层（第{last_layer_idx}层）聚类热图.png')
plt.tight_layout()
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"聚类热图已保存至: {heatmap_path}")
print("所有分析完成!")
