from prepare_data import *


# 测试模型
def test_model(model_class, rank, test_data):
    model = model_class(rank).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for input_data, label in test_data:
            input_ids = input_data['input_ids'].squeeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attention_mask = input_data['attention_mask'].squeeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            output = model(input_ids, attention_mask)
            predicted_label = (output > 0.5).float().item()  # 获取预测结果
            predictions.append(predicted_label)
            labels.append(label)

    # 计算准确率
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(labels)
    print(f'Test Accuracy: {accuracy:.4f}')


# 加载最佳参数并进行测试
with open('best_params.pkl', 'rb') as f:
    best_rank, best_lr, best_batch_size = pickle.load(f)

test_model(BERTSentiment, best_rank, test_data)