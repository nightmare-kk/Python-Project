import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchtext.data import Field, BucketIterator
from transformers import BertModel, BertTokenizer
import random
import loralib as lora
import itertools
import matplotlib.pyplot as plt


# 随机种子
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# 导入数据
# 加载IMDB数据集
dataset = datasets.load_dataset("stanfordnlp/imdb")
train_data = dataset['train']
test_data = dataset['test']

# 定义字段
def preprocess_data(data):
    return [(item['text'], item['label']) for item in data]

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

print(train_data[0])

# 创建BERT的tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization
def tokenize_data(data):
    return [(tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt'), label) for text, label in data]

train_data = tokenize_data(train_data)
test_data = tokenize_data(test_data)

# 创建迭代器
BATCH_SIZE = 16


# 定义BERT模型
class BERTSentiment(nn.Module):
    def __init__(self, rank):
        super(BERTSentiment, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # LoRA适配
        self.bert = lora.Lora(self.bert, rank=rank)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取[CLS]的输出
        return self.sigmoid(self.fc(cls_output))


# 训练模型
def train(model, data, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for input_data, label in data:
        input_ids = input_data['input_ids'].squeeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        attention_mask = input_data['attention_mask'].squeeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        labels = torch.tensor(label).float().unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data)

# 评估模型
def evaluate(model, data, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0

    with torch.no_grad():
        for input_data, label in data:
            input_ids = input_data['input_ids'].squeeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attention_mask = input_data['attention_mask'].squeeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = torch.tensor(label).float().unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            # 计算正确预测的数量
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()

    return epoch_loss / len(data), correct / len(data)

# 交叉验证
def cross_validate(model_class, rank, lr, batch_size, n_folds=5):
    fold_size = len(train_data) // n_folds
    indices = list(range(len(train_data)))
    random.shuffle(indices)

    for fold in range(n_folds):
        print(f"Fold {fold + 1}/{n_folds}")
        val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

        train_subset = [train_data[i] for i in train_indices]
        val_subset = [train_data[i] for i in val_indices]

        # 实例化模型
        model = model_class(rank).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 训练过程
        N_EPOCHS = 5

        for epoch in range(N_EPOCHS):
            train_loss = train(model, train_subset, optimizer, criterion)
            val_loss, val_accuracy = evaluate(model, val_subset, criterion)

            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}')

    # 绘制损失曲线
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



# 网格搜索超参数
def grid_search():
    ranks = [1, 4, 8, 16]
    lrs = [1e-5, 1e-4, 1e-3]
    batch_sizes = [16, 32, 64]

    best_val_loss = float('inf')
    best_params = None

    for rank, lr, batch_size in itertools.product(ranks, lrs, batch_sizes):
        print(f"Training with Rank: {rank}, Learning Rate: {lr}, Batch Size: {batch_size}")
        val_loss = cross_validate(BERTSentiment, rank, lr, batch_size)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (rank, lr, batch_size)

    print("Best Parameters:", best_params)

# 进行网格搜索
grid_search()
