import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv') #엑셀 데이터를 다룸 pandas
#print(data)
'''
데이터 전처리 엑셀 파일에서 비어있는 값들을 찾아서 처리
'''
#data.isnull().sum() #빈값이 있는곳을 찾아줌
data = data.dropna() #비어있는 값을 지워줌

#data.fillna(100) #빈값을 100으로 채워준다
#data['gre'] # 해당열
#data['gre'].min(), ..count()

yData = data['admit'].values #admit열의 값을 저장
xData = []

for i, rows in data.iterrows(): #data라는 dataframe을 가로 한줄씩
    xData.append([ rows['gre'], rows['gpa'], rows['rank'] ])

#신경망 레이어를 쉽게 만들어줌
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'tanh'), #해당 레이어의 노드 개수 관습적으로 2의 제곱수로 표현함, 활성함수
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'), #sigmoid는 0과 1 사이의 값을 알수 있다.. 그것은 확률로 해석가능
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
#binary_crossentropy 오차 함수, 결과가 0과 1사이의 분류/확률 문제에서 씀
model.fit(np.array(xData), np.array(yData), epochs = 1000) #데이터는 numpy나 tensor형태로 넣어야한다
'''
첫번째 인자값들로 두번째 인자를 예측
epochs = 10
데이터셋을 10번 돌면서 학습을 함
'''

#예측
result = model.predict([ [750, 3.70, 3], [400, 2.20, 1] ])
print(result)