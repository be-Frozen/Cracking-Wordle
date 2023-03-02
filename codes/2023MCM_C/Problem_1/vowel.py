import pandas as pd

# 读取CSV文件
df = pd.read_csv('Problem_C_Data_Wordle_test.CSV')

# 定义元音字母集合
vowels = set(['a', 'e', 'i', 'o', 'u'])

# 遍历单词并计算元音字母的个数
for i, word in enumerate(df['Word']):
    vowel_count = 0
    for c in word:
        if c.lower() in vowels:
            vowel_count += 1
    df.loc[i, 'vowel'] = vowel_count

# 将结果保存回CSV文件
df.to_csv('word_list_with_vowels.csv', index=False)