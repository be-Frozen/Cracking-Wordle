import pandas as pd

# 读取CSV文件
df = pd.read_csv('Problem_C_Data_Wordle.CSV')
df['Letter Frequency Sum'] = 0

letter_freq_df = pd.read_csv('letter_frequency.csv')
letter_freq_dict = dict(zip(letter_freq_df['Letter'], letter_freq_df['Frequency']))
for i, row in df.iterrows():
    word = row['Word']
    freq_sum = sum(letter_freq_dict.get(letter, 0) for letter in word)
    df.loc[i, 'letter_frequency_sum'] = freq_sum

df.to_csv('words_with_letter_frequency_sum.csv', index=False)