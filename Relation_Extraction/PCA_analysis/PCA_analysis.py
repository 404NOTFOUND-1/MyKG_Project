import re

import jieba
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例中文文档
# documents = [
#     "根据单体电池封装形式不同，可以分为圆柱电芯、方形电芯及软包电芯。",
#     "电池单体模块将一定数量的单体电池连接在一起。",
#     "电池包在模组的基础上装配电池管理系统、电池冷却系统、线束、支架等零部件。",
#     "电池单体模块的组成包括单体电池、母线牌、隔层、散热铜管、隔离板、BMU采集板、熔丝、温度传感器、采样电压采集线束。",
#     "电池包内包括了电池管理系统、电池冷却系统、电池单体模块、高压系统、上箱体总成、下箱体总成、防火材料总成。",
#     "电池单体模块内每一节单体电池两端均设有保险装置,每个电池片和电池砖都有保险装置。",
#     "电池包还包含B+接触器和B-接触器、电流测量分流器和630A的保险丝。",
#     "高压系统包括加热继电器、加热保险、总负继电器、总正继电器、预充电阻、预充继电器、电池管理系统。",
#     "电池管理系统包括了继电器、电池信息采集器、状态计算和能量管理系统、安全管理系统、热管理系统、接口。",
#     "电池管理系统包括了电池管理器、熔断器、手动维修开关（MSD）、接触器、霍尔传感器/分流器、绝缘模块。",
#     "单体电池按电解质分为酸性电池、碱性电池、中性电池、有机电解质电池、固体电解质电池。"
# ]
documents = []
with open('outputdta.txt', 'r', encoding='utf8') as f:
    for line in f:
        documents.append(line)

# 定义中文停用词列表
stopwords = {'的', '了', '和', '是', '在', '有', '就', '不', '人', '都', '一个', '上', '中', '大', '为', '等', '均'}


def preprocess(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 使用jieba进行分词
    words = jieba.lcut(text)
    # 去除停用词
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


# 预处理文档
preprocessed_docs = [preprocess(doc) for doc in documents]
print(preprocessed_docs)

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_docs)

# 设置聚类数量k
k = 3

# 初始化K-means模型并进行聚类
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_
print(labels)

# 使用PCA将特征降到2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# 绘制聚类结果
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
for i, doc in enumerate(preprocessed_docs):
    plt.annotate(i, (X_reduced[i, 0], X_reduced[i, 1]))
plt.legend('')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clustering of Chinese Text Data')
plt.show()
