import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key='sk-zk2b1f3a61aa82cee0eabc045237fd0bbe866cd9cc4db8f6', base_url='https://api.zhizengzeng.com/v1')

# 1. 读取 CSV 文件
edges = pd.read_csv('/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/edges.csv')

# 2. 确保边类型列存在
print(edges.columns)  # 确认是否有 '边类型' 列
edge_types = edges['边类型'].tolist()

# 3. 获取每个边类型的嵌入，使用 text-embedding-ada-002
def get_embedding(text):
    if isinstance(text, str) and text.strip():  # 检查是否为非空字符串
        response = client.embeddings.create(
            input=[text],  
            model="text-embedding-ada-002",
            encoding_format="float"
        )
        return response.data[0].embedding
    else:
        print(f"Invalid input: {text}")
        return None

# 4. 生成每个边类型的嵌入
embeddings = np.array([get_embedding(text) for text in edge_types])

# 5. 使用肘部法则计算不同簇数的 Inertia
inertia = []
K = range(1, 21)  # 可以调整簇数的范围
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

# 6. 使用 K-means 聚类（根据肘部法则选择合适的簇数）
n_clusters = 20  # 可以根据肘部图选择合适的簇数
kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000)
edges['Cluster'] = kmeans.fit_predict(embeddings)

# 7. 查看聚类结果
print(edges.groupby('Cluster')['边类型'].unique())

# 8. 根据聚类结果合并相似的边类型
# 选择每个聚类中频率最高的边类型作为代表
cluster_representative = edges.groupby('Cluster')['边类型'].agg(lambda x: x.mode().iloc[0]).to_dict()
edges['合并后的边类型'] = edges['Cluster'].map(cluster_representative)

# 9. 去除聚类后仍然重复的边（基于源节点、边类型和目标节点）
edges_final = edges.drop_duplicates(subset=['源节点', '合并后的边类型', '目标节点'])

# 10. 保存最终整合后的 CSV 文件
final_csv_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/clustered_edges.csv'
edges_final.to_csv(final_csv_path, index=False)

# 11. 打印检查结果
print(edges_final.head())

# 12. 使用 PCA 降维并可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=edges['Cluster'], cmap='viridis', alpha=0.6)
plt.title('K-Means Clustering Visualization with Ada-002 Embeddings and PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
# 保存 PCA 可视化结果
pca_visualization_path = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/ada_pca_clustering_visualization.png'
plt.savefig(pca_visualization_path, format='png', dpi=300)
plt.show()
