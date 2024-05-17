import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

from data_process import bilstm_data_process

# 准备数据
# sentences = [
#     "Marie Curie was born in Warsaw, discovered radium, and died in Passy.",
#     "Albert Einstein was born in Ulm, developed the theory of relativity, and died in Princeton.",
#     "Nikola Tesla was born in Smiljan, invented the Tesla coil, and died in New York.",
#     "Isaac Newton was born in Woolsthorpe, formulated the laws of motion, and died in London.",
#     "Galileo Galilei was born in Pisa, improved the telescope, and died in Arcetri."
# ]
# labels = [
#     ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O", "O", "O", "B-LOC",
#      "O"],
#     ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O", "O", "O", "B-LOC",
#      "O"],
#     ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O", "O", "O", "B-LOC",
#      "O"],
#     ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O", "O", "O", "B-LOC",
#      "O"],
#     ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O", "O", "O", "B-LOC", "O"]
# ]
sentences = []
labels = []
datas = bilstm_data_process('all.jsonl', 'doccano_conv.json')
for data in datas:
    print(data[0], data[1])
    sentences.append(data[0])
    labels.append(data[1])

# 标签编码
label_encoder = LabelEncoder()
label_encoder.fit(["O", "B-PAR", "I-PAR", "B-TYPE", "I-TYPE", "B-MAT", "I-MAT"])
y = [label_encoder.transform(label) for label in labels]

# 词汇表和句子填充
words = list(set([word for sentence in sentences for word in sentence]))
words.append("PAD")
num_words = len(words)
word2idx = {w: i for i, w in enumerate(words)}
max_len = 30
X = [[word2idx[w] for w in sentence] for sentence in sentences]
X = [x + [word2idx["PAD"]] * (max_len - len(x)) for x in X]
y = [list(label) + [label_encoder.transform(["O"])[0]] * (max_len - len(label)) for label in y]


# 数据集定义
class RelationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


# 模型定义
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=50):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.hidden2tag(x)
        return x

    def loss(self, feats, tags, mask):
        return -self.crf(feats, tags, mask=mask)

    def decode(self, feats, mask):
        return self.crf.decode(feats, mask=mask)


# 数据加载
dataset = RelationDataset(X, y)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 模型训练
vocab_size = num_words
tagset_size = len(label_encoder.classes_)
model = BiLSTM_CRF(vocab_size, tagset_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()
        mask = (X_batch != word2idx["PAD"]).to(torch.uint8)
        mask[:, 0] = 1  # 确保第一个时间步的mask值为1
        feats = model(X_batch)
        loss = model.loss(feats, y_batch, mask=mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

# 关系预测
model.eval()
test_sentence = "Nikola Tesla was born in Smiljan, invented the Tesla coil, and died in New York."
test_X = [[word2idx.get(w, word2idx["PAD"]) for w in test_sentence.split()]]
test_X = [x + [word2idx["PAD"]] * (max_len - len(x)) for x in test_X]
test_X = torch.tensor(test_X, dtype=torch.long)

with torch.no_grad():
    mask = (test_X != word2idx["PAD"]).to(torch.uint8)
    mask[:, 0] = 1  # 确保第一个时间步的mask值为1
    feats = model(test_X)
    predictions = model.decode(feats, mask=mask)
pred_labels = label_encoder.inverse_transform(predictions[0])
print("Predicted labels:", pred_labels)
