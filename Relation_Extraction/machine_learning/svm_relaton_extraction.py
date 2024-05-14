import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sentences = []
relations = []
entities = []
new_sentences = []
new_entities = []
with open('output.json', 'r', encoding='utf8') as f:
    i = 0
    for line in f:
        data = json.loads(line)
        sentences.append(data['text'])
        relations.append(data['relations'])
        entities.append(data['entities'])
        i += 1

sentences, new_sentences, relations, new_relations, entities, new_entities = \
    train_test_split(sentences, relations, entities, test_size=0.4, random_state=412)

# 特征提取：将句子转换为词袋模型特征
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(relations)
new_y = label_encoder.fit_transform(new_relations)
# 构建SVM模型
svm_model = make_pipeline(CountVectorizer(), SVC(kernel='linear', probability=True))

# 模型训练
svm_model.fit(sentences, y)

# 模型评估

new_y_pred = svm_model.predict(new_sentences)
predicted_relations = label_encoder.inverse_transform(new_y_pred)
print(classification_report(new_y, new_y_pred, target_names=label_encoder.classes_))

for sent, entities, relation in zip(new_sentences, new_entities, predicted_relations):
    print(f"Sentence: {sent}")
    print(f"Entities: {entities}")
    print(f"Predicted Relation: {relation}")
    print()
