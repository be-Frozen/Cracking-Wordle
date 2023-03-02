import pandas as pd

# 读取CSV文件
df = pd.read_csv('Problem_C_Data_Wordle.CSV')

# 添加一列用于存储ASCII码和
df['ASCII Sum'] = 0

# 遍历单词并计算ASCII码和
for i, word in enumerate(df['Word']):
    ascii_sum = 0
    for c in word:
        ascii_sum += ord(c)
    # 在新列中保存ASCII码和
    df.loc[i, 'ASCII Sum'] = ascii_sum

# 将结果保存回CSV文件
df.to_csv('word_list_with_ascii_sum.csv', index=False)