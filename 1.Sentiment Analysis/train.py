from datasets import load_dataset
import jieba
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
from transformers import Trainer, TrainingArguments
import evaluate


# 读取数据
dataset = load_dataset('csv', data_files='weibo_senti_100k.csv', split='train')

# 数据集划分
dataset = dataset.train_test_split(test_size=0.2)

# 数据集预处理
dataset = dataset.filter(lambda x: x['review'] is not None and x['label'] is not None)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

def process_function(dataset):
    dataset['tokens'] = " ".join(jieba.lcut(dataset['review']))
    tokenized_dataset = tokenizer(dataset['tokens'], max_length=128, padding='max_length', truncation=True)
    tokenized_dataset['labels'] = dataset['label']
    return tokenized_dataset

tokenized_dataset = dataset.map(process_function, remove_columns=dataset['train'].column_names)

tokenized_dataset = tokenized_dataset.with_format('torch')

# 创建模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 创建评估函数
acc_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')

def compute_metrics(eval_predict):
    preds, labels = eval_predict
    preds = preds.argmax(axis=1)
    acc = acc_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels)
    return {
        'accuracy': acc,
        'f1': f1
    }

# 训练模型
train_args = TrainingArguments(
    output_dir='./checkpoints',
    warmup_ratio=0.2,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=3,
    weight_decay=0.01,
    metric_for_best_model='accuracy',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

trainer.train()

# 保存最优模型路径模型
best_ckpt_path = trainer.state.best_model_checkpoint