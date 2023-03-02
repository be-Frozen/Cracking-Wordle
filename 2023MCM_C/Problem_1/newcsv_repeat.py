import pandas as pd

# 读取CSV文件
df = pd.read_csv('Problem_C_Data_Wordle.CSV')

df['Has Duplicate Letters'] = False

# 遍历单词并检查是否包含重复字母
for i, word in enumerate(df['Word']):
    has_duplicate_letters = len(set(word)) != len(word)
    df.loc[i, 'Has Duplicate Letters'] = has_duplicate_letters

# 将结果保存回CSV文件
df.to_csv('word_list_with_duplicates.csv', index=False)