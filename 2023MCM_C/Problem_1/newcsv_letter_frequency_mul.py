import pandas as pd

import numpy as np

# 读取CSV文件
df = pd.read_csv('Problem_C_Data_Wordle_test.CSV')
df['Letter Frequency Sum'] = 0

letter_freq_df = pd.read_csv('letter_frequency.csv')

letter_freq_dict = dict(zip(letter_freq_df['Letter'], -np.log2(letter_freq_df['Frequency'])*letter_freq_df['Frequency']))


# 计算每个单词中各个字母使用频率的乘积，并保存在新的一列中
for i, row in df.iterrows():
    word = row['Word']
    freq_mul = sum(letter_freq_dict.get(letter, 0) for letter in word)
    df.loc[i, 'letter_frequency_mul'] = freq_mul

df.to_csv('words_with_letter_frequency_mul.csv', index=False)