from random import random
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb


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


# 用交叉验证找到最佳的参数
param_test = {
    'num_leaves': range(10, 30, 10),
    'max_depth': range(3, 5, 1),
    'learning_rate': [0.01, 0.05],
    'n_estimators': range(50, 200, 50),
    'min_child_samples': range(10, 30, 10),
    'subsample': [0.6, 0.7],
    'colsample_bytree': [0.6, 0.7],
    'reg_alpha': [0, 0.001, 0.01],
    'reg_lambda': [0, 0.001, 0.01]
}
# 多分类问题，交叉验证，评估指标为准确率
gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(objective='multiclass', num_class=3),
                       param_grid=param_test, scoring='accuracy', cv=5, verbose=50)


gsearch.fit(X_train, y_train)
print(gsearch.best_params_)
print(gsearch.best_score_)
# 预测
y_pred = gsearch.predict(X_test)
# 准确率
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))