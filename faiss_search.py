import faiss
import numpy as np

# 向量维度（如 BERT embedding）
dimension = 1024

# 1️⃣ 创建 Inner Product 索引
# 归一化后，IP = Cosine Similarity
index = faiss.IndexFlatIP(dimension)

# 2️⃣ 构造向量（模拟 embedding）
vectors = np.random.random((1000, dimension)).astype("float32")

# 3️⃣ L2 归一化（⚠️关键步骤）
faiss.normalize_L2(vectors)

# 4️⃣ 添加到索引
index.add(vectors)

# 5️⃣ 构造查询向量
query_vector = np.random.random((1, dimension)).astype("float32")

# 6️⃣ 查询向量也必须归一化
faiss.normalize_L2(query_vector)

# 7️⃣ 相似度搜索
k = 5
scores, indices = index.search(query_vector, k)

print("最相似向量索引:", indices)
print("Cosine 相似度:", scores)