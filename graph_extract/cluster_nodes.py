import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import openai  # 需要安装 OpenAI 的 Python 库
from openai import OpenAI
import os

client = OpenAI(api_key='sk-zk2b1f3a61aa82cee0eabc045237fd0bbe866cd9cc4db8f6'
                ,base_url='https://api.zhizengzeng.com/v1')
# 1. 读取 CSV 文件
df = pd.read_csv('/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/nodes.csv')

# 2. 去除完全重复的节点
df = df.drop_duplicates()

# 3. 保存去重后的 CSV 文件
df.to_csv('/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/nodes_pro.csv', index=False)

# 4. 读取去重后的 CSV 文件
df = pd.read_csv('/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/nodes_pro.csv')

# 5. 获取每个节点类型的嵌入，使用 text-embedding-ada-002
def get_embedding(text):
    # print(text)
    # print(type(text))
    if isinstance(text, str) and text.strip():  # 检查是否为非空字符串
        response = client.embeddings.create(
            input=[text],  
            model="text-embedding-ada-002",
            encoding_format="float"
        )
        # print(response['data'][0]['embedding'])
        return response.data[0].embedding
    else:
        print(f"Invalid input: {text}")
        return None

# 6. 生成每个节点类型的嵌入
embeddings = np.array([get_embedding(text) for text in df['节点类型']])

# 7. 使用肘部法则计算不同簇数的 Inertia
inertia = []
K = range(1, 21)  # 你可以调整簇数的范围
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
# 保存肘部图的路径
elbow_visualization_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/elbow_method_visualization.png'
plt.savefig(elbow_visualization_path, format='png', dpi=300)
plt.show()

# 继续进行后续的 K-means 聚类操作
# 8. 使用 K-means 聚类
n_clusters = 25  # 根据实际情况调整簇的数量，建议根据肘部法则选择
kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000)
df['Cluster'] = kmeans.fit_predict(embeddings)  # 确保 'Cluster' 列存在

# 检查是否成功添加 'Cluster' 列
print(df.head())  # Debugging step to ensure 'Cluster' column exists

# 9. 查看聚类结果并保存每个簇的节点类型到 CSV 文件
# 获取每个簇的节点类型并转换为字符串形式
clustered_nodes = df.groupby('Cluster')['节点类型'].unique().reset_index()
clustered_nodes['节点类型'] = clustered_nodes['节点类型'].apply(lambda x: ', '.join(x))

# 保存聚类结果到指定的 CSV 文件
clustered_nodes_csv_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/clustered_nodes.csv'
clustered_nodes.to_csv(clustered_nodes_csv_path, index=False)
print(f"每个簇的节点类型已保存到 {clustered_nodes_csv_path}")

# 10. 根据聚类结果合并相似的节点
# 选择每个聚类中最常见的节点类型作为代表
cluster_representative = df.groupby('Cluster')['节点类型'].agg(lambda x: x.mode().iloc[0]).to_dict()
df['合并后的类型'] = df['Cluster'].map(cluster_representative)

# 11. 去除聚类后仍然重复的节点（基于节点名称和合并后的类型）
df_final = df.drop_duplicates(subset=['节点名称', '合并后的类型'])

# 12. 选择需要的两列：'节点名称' 和 '合并后的类型'
df_final = df_final[['节点名称', '合并后的类型']]

# 13. 重命名列名为 '节点名称' 和 '节点类型'
df_final = df_final.rename(columns={'合并后的类型': '节点类型'})

# 14. 保存最终整合后的 CSV 文件
final_csv_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/cluster_nodes.csv'
df_final.to_csv(final_csv_path, index=False)

# 15. 打印检查结果
print(df_final.head())

# 16. 使用 PCA 降维并可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('K-Means Clustering Visualization with Ada-002 Embeddings and PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
# 保存 PCA 可视化结果
pca_visualization_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/ada_pca_clustering_visualization.png'
plt.savefig(pca_visualization_path, format='png', dpi=300)
plt.show()

# 17. 使用 t-SNE 降维并可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('K-Means Clustering Visualization with Ada-002 Embeddings and t-SNE')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(scatter, label='Cluster')
# 保存 t-SNE 可视化结果
tsne_visualization_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/ada_tsne_clustering_visualization.png'
plt.savefig(tsne_visualization_path, format='png', dpi=300)
plt.show()
