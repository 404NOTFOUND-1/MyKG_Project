import torch
import torch.nn as nn
import torch.optim as optim


# 简单的BiLSTM模型
class EntityRecognitionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(EntityRecognitionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.log_softmax(tag_space, dim=1)
        return tag_scores


# 参数
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
TAGSET_SIZE = 10  # 比如: 'O', 'TERM', 'PROD', 'ORG', 'PER', 'TIME', 'QUAN'

# 初始化模型、损失函数和优化器
model = EntityRecognitionModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 示例输入数据
sentence = torch.tensor([2, 3, 5, 2, 7, 45, 4, 10, 7, 6], dtype=torch.long)
tags = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)

# 训练模型
for epoch in range(300):
    model.zero_grad()
    tag_scores = model(sentence)
    loss = loss_function(tag_scores, tags)
    loss.backward()
    optimizer.step()

# 测试
with torch.no_grad():
    test_sentence = torch.tensor([45, 7, 10], dtype=torch.long)
    tag_scores = model(test_sentence)
    predicted_tags = torch.argmax(tag_scores, dim=1)
    print(predicted_tags)  # 输出应为最可能的标签序列
