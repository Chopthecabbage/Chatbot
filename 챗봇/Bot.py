#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# In[2]:


import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ### Tokenizer는 문장으로부터 단어를 토큰화하고 숫자에 대응시키는 딕셔너리를 사용할 수 있도록 합니다.
# → 느낌표, 마침표와 같은 구두점은 인코딩에 영향을 주지 않습니다.
# 
# ### pad_sequences → 길이가 같지 않고 적거나 많을 때 일정한 길이로 맞춰 줄 때 사용한다.

# In[3]:


with open('OriginalChatbotDataset.json') as file:
    data = json.load(file)


# In[4]:


from sklearn.preprocessing import LabelEncoder


# ### LabelEncoder → 문자를 숫자(수치화), 숫자를 문자로 매핑

# In[5]:


sentences = [] # 패턴 문장들 
labels = []    # 패턴 라벨들
classes = []   # 태그 종류들
responses = [] # 응답 문장들

i = 0
for intent in data['intents']: 
    if i < 1000:
        sentences.append(intent['patterns'])
        labels.append(intent['tag'])
            
        responses.append(intent['responses'])
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
        i += 1
        
    elif i > 999:
        for pattern in intent['patterns']:
            sentences.append(pattern)
            labels.append(intent['tag'])

        responses.append(intent['responses'])

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[6]:


encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels) # → 피팅하고 라벨숫자로 변환한다.


# In[7]:


vocab_size = 10000

# 임베딩 벡터 차원
embedding_dim = 256

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 20
trunc_type = 'post'

# 태그 단어 
OOV = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=OOV) # adding out of vocabulary token
tokenizer.fit_on_texts(sentences) # 문자 데이터를 입력받아서 리스트의 형태로 변환합니다.
word_index = tokenizer.word_index # 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환합니다.
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, truncating=trunc_type, maxlen=max_sequences) # 시퀀스의 최대길이를 20으로 지정하고 초과한 경우 각 시퀀스의 뒤쪽에서 자른다.


# ### Tokenizer의 oov_token 인자를 사용하면 미리 인덱싱하지 않은 단어들은 ‘<OOV>’로 인덱싱됩니다.

# In[8]:


model = tf.keras.models.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_sequences))
# model.add(keras.layers.Flatten())
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(len(classes), activation='softmax'))


# 1. Flatten 레이어 대신, GlobalAveragePooling1D 레이어를 사용할 수 있습니다. → 훈련 과정이 조금 더 빨라지는 대신, 정확도가 조금 감소합니다.
# 2. GlobalAveragePooling1D 해당 레이어에서는 각 예시에 대해 sequence 차원을 평균하여 고정된 길이의 벡터를 출력한다. → 이를 통해 가변적인 길이의 입력을 간단하게 처리할 수 있다.

# In[9]:


model.summary()


# In[10]:


training_labels = np.array(labels)


# In[ ]:


EPOCHS = 500
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(padded, training_labels, epochs=EPOCHS)


# sparse_categorical_crossentropy → 다중 분류 손실함수. 위와 동일하지만, integer type 클래스라는 것이 다르다.
# 예를 들면 출력 실측값이 아래와 같은 형태로 one-hot encoding 과정을 하지 않아도 된다. 
# 
# 출처: https://crazyj.tistory.com/153 [크레이지J의 탐구생활]

# In[ ]:


def check():
    print('start talking with bot, Enter quit to exit')
    while True:
        string = input('Enter: ')
        if string == 'quit': break
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([string]),
                                             truncating=trunc_type, maxlen=max_sequences))

        category = encoder.inverse_transform([np.argmax(result)]) # 인덱스(라벨숫자)를 입력하면 본래 값을 반환
        for i in data['intents']:
            if i['tag'] == category:
                compare = category.astype('int')
                if compare < 1000:
                    print(i['responses'])
                else:
                    print(np.random.choice(i['responses'])) 


# In[ ]:


check()


# In[ ]:


tf.keras.models.save_model(model, "형스비")

