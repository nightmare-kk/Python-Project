import pandas as pd

# 创建示例 DataFrame
data = {
    'A': ['cat', 'dog', 'cat', 'bird'],
    'B': [1, 2, 3, 4]
}
df = pd.DataFrame(data)

# 将列 'A' 进行 one-hot 编码
df_one_hot = pd.get_dummies(df, columns=['A'])

df['A'] = df['A'].factorize()[0]

print(df_one_hot)

# 只读取csv文件的第一列进dataframe
