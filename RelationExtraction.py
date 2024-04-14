import torch
import torch.nn as nn
import torch.optim as optim


# BiLSTM+Attention模型
class RelationExtractionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, relation_size):
        super(RelationExtractionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.relation_fc = nn.Linear(hidden_dim * 2, relation_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        attention_weights = torch.tanh(self.attention(lstm_out))
        attention_weights = torch.softmax(attention_weights, dim=0)
        context = lstm_out * attention_weights
        context = context.sum(dim=0)
        relation_scores = self.relation_fc(context)
        return torch.log_softmax(relation_scores, dim=1)


# 参数
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
RELATION_SIZE = 5  # 如 'is-a', 'part-of', 'same-as', 'has-a', 'none'

# 初始化模型、损失函数和优化器
model = RelationExtractionModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, RELATION_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 示例输入数据
sentence = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
relation_label = torch.tensor([0], dtype=torch.long)

# 训练模型
for epoch in range(300):
    model.zero_grad()
    relation_scores = model(sentence)
    loss = loss_function(relation_scores, relation_label)
    loss.backward()
    optimizer.step()

# 测试
with torch.no_grad():
    test_sentence = torch.tensor([1, 2], dtype=torch.long)
    relation_scores = model(test_sentence)
    predicted_relation = torch.argmax(relation_scores, dim=1)
    print(predicted_relation)  # 输出应为最可能的关系类型
