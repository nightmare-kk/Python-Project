from random import random
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score


# 示例数据
np.random.seed(0)
X = np.random.rand(1000, 36)  # 1000 行，36 列特征
y = np.random.randint(0, 3, size=1000)  # 3 类标签

# 创建 DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(36)])
df['label'] = y

# 分割数据集为训练集和测试集
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 导入真实数据
# processed_data = pd.read_csv('processed_data.csv')
# X = processed_data.drop('sample_flag', axis=1)
# y = processed_data['sample_flag'] - 1
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义XGBoost参数空间
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}


# 创建XGBoost分类器
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    tree_method='hist'  # 启用加速
)


# 使用RandomizedSearchCV进行超参数搜索
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,  # 选择 50 种参数组合进行测试
    scoring='accuracy',
    cv=5,  # 5 折交叉验证
    verbose=1,
    random_state=42,
    n_jobs=-1  # 使用所有可用的 CPU 核心
)


# 进行参数搜索并保存最佳参数和模型
random_search.fit(X_train, y_train)
print(f'Best Parameters: {random_search.best_params_}')
print(f'Best Score: {random_search.best_score_}')



# # 记录训练和验证损失
# train_losses = []
# val_losses = []
#
# # 使用得到的最佳参数进行训练，训练总轮数为500轮，每10轮输出一次训练损失和验证损失
# best_params = random_search.best_params_
# best_model = xgb.XGBClassifier(**best_params)
# history = best_model.fit(
#     X_train, y_train,
#     eval_set=[(X_train, y_train), (X_test, y_test)],
#     eval_metric='logloss',
#     verbose=10,
#     early_stopping_rounds=10,
#     evals_result={'train': {}, 'validation': {}}
# )


# # 获取训练和验证损失
# train_losses = history.history['train']['logloss']
# val_losses = history.history['validation']['logloss']
#
# # 绘制训练和验证损失图
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(train_losses)), train_losses, label='Training Loss', color='blue')
# plt.plot(np.arange(len(val_losses)), val_losses, label='Validation Loss', color='orange')
# plt.title('Training and Validation Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.show()

# # 使用训练好的模型进行验证
# preds = best_model.predict(X_test)  # 在测试集上进行预测
# accuracy = accuracy_score(y_test, preds)  # 计算准确率
# print(f'Test Accuracy: {accuracy:.2f}')  # 输出测试集准确率