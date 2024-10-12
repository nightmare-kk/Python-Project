import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
import random
import itertools
import matplotlib.pyplot as plt
import pickle
from peft import get_peft_model, LoraConfig, TaskType

# 设置随机种子
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

# 加载IMDB数据集
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

# 定义字段
def preprocess_data(data):
    return [(item['text'], item['label']) for item in data]

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 创建BERT的tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization
def tokenize_data(data):
    encodings = tokenizer([text for text, _ in data], padding=True, truncation=True, max_length=512, return_tensors='pt')
    return [(encodings, label) for (_, label) in data]

train_data = tokenize_data(train_data)
test_data = tokenize_data(test_data)

# 创建迭代器
BATCH_SIZE = 2  # 减小批量大小

# 定义BERT模型
class BERTSentiment(nn.Module):
    def __init__(self, rank):
        super(BERTSentiment, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

        # LoRA配置
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type=TaskType.SEQ_CLS
        )

        # 应用LoRA适配
        self.bert = get_peft_model(self.bert, lora_config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # 取最后的logits

# 训练模型
def train(model, data, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for input_data, label in data:
        input_ids = input_data['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        attention_mask = input_data['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        labels = torch.tensor(label).float().unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 清理未使用的缓存
        torch.cuda.empty_cache()

    return epoch_loss / len(data)

# 评估模型
def evaluate(model, data, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0

    with torch.no_grad():
        for input_data, label in data:
            input_ids = input_data['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attention_mask = input_data['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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

    train_losses = []
    val_losses = []

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

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}')

    # 绘制损失曲线
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # 返回最后一个验证损失
    return val_losses[-1]

# 网格搜索超参数
def grid_search():
    ranks = [1, 4, 8, 16]
    lrs = [1e-5, 1e-4, 1e-3]
    batch_sizes = [8]  # 只使用一个批量大小，避免过多的组合

    best_val_loss = float('inf')
    best_params = None

    for rank, lr, batch_size in itertools.product(ranks, lrs, batch_sizes):
        print(f"Training with Rank: {rank}, Learning Rate: {lr}, Batch Size: {batch_size}")
        val_loss = cross_validate(BERTSentiment, rank, lr, batch_size)

        # 这里可以根据验证集的最后一个损失来决定是否更新最佳参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (rank, lr, batch_size)

    # 保存最佳参数
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    print("Best Parameters:", best_params)


# 进行网格搜索
grid_search()
