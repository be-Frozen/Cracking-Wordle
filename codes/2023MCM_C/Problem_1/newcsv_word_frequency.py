import pandas as pd

# 读取 Problem_C_Data_Wordle.CSV 文件
df_wordle = pd.read_csv('Problem_C_Data_Wordle.CSV')

# 读取 words_frequency.csv 文件
df_word_freq = pd.read_csv('words_frequency.csv')

# 合并两个 DataFrame，使用 Word 列作为关键字
df_merged = pd.merge(df_wordle, df_word_freq, on='Word', how='left')

# 选择 Value 列并将其添加为新列
df_wordle['Value'] = df_merged['Value']

# 保存结果
df_wordle.to_csv('Problem_C_Data_Wordle_with_Value.csv', index=False)