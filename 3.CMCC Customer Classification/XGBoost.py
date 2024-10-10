import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score


# 导入数据
processed_data = pd.read_csv('processed_data.csv')
X = processed_data.drop('sample_flag', axis=1)
y = processed_data['sample_flag'] - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='gpu_hist'  # 启用 GPU 加速
)

# 使用RandomizedSearchCV进行参数搜索
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

# 训练模型并添加早期停止
def fit_with_early_stopping(model, X_train, y_train, X_val, y_val):
    evals = [(X_train, y_train), (X_val, y_val)]
    history = model.fit(X_train, y_train,
              eval_set=evals,
              early_stopping_rounds=10,  # 10 轮没有提升则停止
              verbose=False)  # 不输出详细信息
    return model, history

# 进行参数搜索
best_score = 0
best_params = None

# 记录训练和验证损失
train_losses = []
val_losses = []

for param_combination in random_search.rvs(50):
    # 设置参数
    xgb_model.set_params(**param_combination)

    # 进行训练和验证
    xgb_model, history = fit_with_early_stopping(xgb_model, X_train, y_train, X_test, y_test)

    # 记录损失
    train_losses.append(history['train']['mlogloss'])
    val_losses.append(history['validation']['mlogloss'])

    # 计算准确率
    preds = xgb_model.predict(X_test)
    score = accuracy_score(y_test, preds)

    # 记录最佳参数和得分
    if score > best_score:
        best_score = score
        best_params = param_combination

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# 绘制训练和验证损失图
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(train_losses)), train_losses, label='Training Loss', color='blue')
plt.plot(np.arange(len(val_losses)), val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 使用最佳参数进行预测
best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', tree_method='gpu_hist')
best_model.fit(X_train, y_train)

# 计算并输出测试集准确率
preds = best_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f'Test Accuracy: {accuracy:.2f}')

# 预测并输出准确率
predict_data = pd.read_csv('predict_data.csv')

preds = best_model.predict(predict_data)
accuracy = accuracy_score(y_test, preds)
print(f'Test Accuracy: {accuracy:.2f}')