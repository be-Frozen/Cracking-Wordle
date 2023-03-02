import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
data = pd.read_csv('Problem_C_Data_Wordle_hello.csv')

# 将输入和输出变量分离
#X = data[['words_frequency', 'ASCII Sum', 'Has Duplicate Letters', 'letter_frequency_mul', 'letter_frequency_sum']]
X = data[['letter_frequency_mul', 'vowel', 'words_frequency_log', 'words_frequency', 'Has Duplicate Letters']]

#print(X)
y = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']]
list = y.values.tolist()
#print(list)
n=356
# 一共358行 0~358
for i in range(n):
    sum=0
    for j in range(7):
        sum=sum+list[i][j]
    for j in range(7):
        list[i][j]=list[i][j]/sum
#print(list)
#y=list
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

np.set_printoptions(precision=5,suppress=False)

# 构建和训练MLP模型
model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=10000)
#y_train = np.apply_along_axis(softmax, 1, y_train)
model.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = model.predict(X_test)
#y_pred = np.apply_along_axis(softmax, 1, y_pred)
#y_test = np.apply_along_axis(softmax, 1, y_test)

#print(y_pred)
print('\n')
y_test = y_test.values.tolist()
#print(y_test)
print('\n')

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

# 归一化

def normalize(y):
    # 将 y 转换为 NumPy 数组
    y = np.array(y,dtype=np.float32)
    # 对每组数据进行百分比转化为和为1的处理
    for i in range(y.shape[0]):
        # 获取当前组的数据
        data = y[i]
        # 计算数据的总和
        total = np.sum(data)
        # 将数据转换为百分比
        percentages = data / total

        # 将百分比转换为和为1的数据
        normalized_data = percentages / np.sum(percentages)
        # 将处理后的数据赋值回 y
        y[i] = normalized_data
        print(y[i])
    return y

y_pred_normalized = normalize(y_pred)
y_test_normalized = normalize(y_test)

print(y_pred_normalized)
print('\n')
print(y_test_normalized)
mse = mean_squared_error(y_test_normalized, y_pred_normalized)
print('MSE:', mse)

word_to_predict='eerie'
vowels = {'a', 'e', 'i', 'o', 'u'}
letter_freq_df = pd.read_csv('letter_frequency.csv')
words_freq_df=pd.read_csv('words_frequency.csv')
letter_freq_dict = dict(zip(letter_freq_df['Letter'], -np.log2(letter_freq_df['Frequency'])))
word_freq_dict = dict(zip(words_freq_df['Word'], words_freq_df['Value']))
word_letter_frequency_mul=0.0
vowel_count=0;
word_freq=0.0;
word_freq_log=0.0;
has_duplicate_letters=False

for letter in word_to_predict:
    word_letter_frequency_mul += letter_freq_dict.get(letter)
    if letter.lower() in vowels:
        vowel_count += 1
    word_freq=word_freq_dict.get(word_to_predict)
    word_freq_log=-np.log2(word_freq)
    has_duplicate_letters = len(set(word_to_predict)) != len(word_to_predict)

print(vowel_count)
print(word_letter_frequency_mul)
print(word_freq_log)
print(word_freq)
print(has_duplicate_letters)

word_attribution=[[word_letter_frequency_mul,vowel_count,word_freq_log,word_freq,has_duplicate_letters]]
print(word_attribution)

y_result=model.predict(word_attribution)
print(y_result)
