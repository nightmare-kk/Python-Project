import pandas as pd
import numpy as np


# 导入数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
full_data = pd.merge(train_data, test_data, how='outer')

train_data.info()
test_data.info()
full_data.info()

# 数据预处理

# 去掉无意义列
drop_columns = ['user_id', 'term_brand', 'term_price', 'change_equip_period_avg', 'join_date']
full_data.drop(drop_columns, axis=1, inplace=True)

# 字符串列分类编码
factorize_columns = ['zfk_type', 'jt_5gwl_flag']
full_data['zfk_type'] = full_data['zfk_type'].factorize()[0]
full_data['jt_5gwl_flag'] = full_data['jt_5gwl_flag'].factorize()[0]
full_data['jt_5gzd_flag'] = full_data['jt_5gzd_flag'].factorize()[0]
full_data['avg3_llb_flag'] = full_data['avg3_llb_flag'].factorize()[0]


# 填充缺失值
missing_columns = full_data.isnull().any()
for col in missing_columns[missing_columns].index:
    full_data[col].fillna(full_data[col].mean(), inplace=True)

# area_code整体值过大，缩小范围
full_data['area_code'] -= full_data['area_code'].min()

# 调整数据类型
full_data['age'] = full_data['age'].astype('float64')
full_data['sl_flag'] = full_data['sl_flag'].astype('int64')
full_data['sl_type'] = full_data['sl_type'].astype('int64')

# 归一化
normal_columns = train_data.select_dtypes(include='float64').columns
for col in normal_columns:
    full_data[col] = (full_data[col] - full_data[col].mean()) / full_data[col].std()

# 保存
full_data.to_csv('processed_data.csv', index=False)