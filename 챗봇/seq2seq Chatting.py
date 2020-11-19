#!/usr/bin/env python
# coding: utf-8

# # 한글 챗봇 딥러닝 프로그램

# In[1]:


from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from konlpy.tag import Okt


# In[2]:


# 태그 단어 
# 이 단어들이 seq2seq의 동작을 제어합니다. → 디코더 입력에 START가 들어가면 디코딩의 시작을 의미합니다. 반대로 디코더 출력에 END가 나오면 디코딩을 종료합니다.
PAD = "<PADDING>" # 패딩
STA = "<START>" # 시작
END = "<END>" # 끝
OOV = "<OOV>" # 없는 단어

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT = 0
DECODER_INPUT = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 30

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든레이어 차원
lstm_hidden_dim = 128

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 챗봇 데이터 로드
chatbot_data = pd.read_csv('./seq2seq_chatbot-master/dataset/chatbot/ChatbotData.csv', encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])


# 참고할 자료 → https://github.com/nawnoes/WellnessConversationAI, https://github.com/deepseasw/seq2seq_chatbot
# 송영숙님이 공개한 한글 데이터셋 → https://github.com/songys/Chatbot_data

# In[3]:


# 데이터 개수
print("데이터 개수: %d"%len(question))


# In[4]:


# 데이터의 일부만 학습에 사용
question = question[:10]
answer = answer[:10]

# 챗봇 데이터 출력
for i in range(5):
    print('Q: ' + question[i])
    print('A: ' + answer[i] + '\n')


# # 단어 사전 생성

# In[5]:


# 형태소 분석 함수 → 문장을 먼저 최소 단위인 토큰으로 나누어야 합니다.
def pos_tag(sentences):
    
    # KoNLPy 형태소분석기 설정
    tagger = Okt()
    
    # 문장 품사 변수 초기화
    sentences_pos = []
    
    # 모든 문장 반복
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)
        
        # 배열인 형태소 분석의 출력을 띄어쓰기로 구분하여 붙임
        sentence = " ".join(tagger.morphs(sentence)) # morphs(phrase) → Parse phrase to morphemes.
        sentences_pos.append(sentence)

    return sentences_pos


# # line 13
# sentence = "a".join(tagger.morphs(sentence)) → 하루a가a또a가네요

# In[6]:


# 형태소분석 수행
question = pos_tag(question)
answer = pos_tag(answer)

# 형태소 분석으로 변환된 챗봇 데이터 출력
for i in range(5):
    print('Q: ' + question[i])
    print('A: ' + answer[i] + '\n')


# In[7]:


# 질문과 대답 문장들을 하나로 합침
sentences = []
sentences.extend(question)
sentences.extend(answer)

words = []

# 단어들의 배열 생성
for sentence in sentences:
    for word in sentence.split(): # a.split() 처럼 괄호 안에 아무 값도 넣어 주지 않으면 공백(스페이스, 탭, 엔터 등)을 기준으로 문자열을 나누어 준다. 
        words.append(word)

# 길이가 0인 단어는 삭제
words = [word for word in words if len(word) > 0] # 리스트 표현식 → 반복문

# 중복된 단어 삭제
words = list(set(words))

# 제일 앞에 태그 단어 삽입
words[:0] = [PAD, STA, END, OOV]


# In[8]:


# 단어 개수
print("단어 개수: %d"%len(words))


# In[9]:


# 단어 출력
print("단어 출력: %s"%words[:70])


# In[10]:


# 단어와 인덱스의 딕셔너리 생성
word_to_index = {word: index for index, word in enumerate(words)} # enumerate → 인덱스와 값이 같이 출력
index_to_word = {index: word for index, word in enumerate(words)}


# In[11]:


# 단어 -> 인덱스
# 문장을 인덱스로 변환하여 모델 입력으로 사용
dict(list(word_to_index.items())[:70])


# In[12]:


# 인덱스 -> 단어
# 모델의 예측 결과인 인덱스를 문장으로 변환시 사용
dict(list(index_to_word.items())[:70])


# # 데이터 전처리
# : 분석하기 좋게 데이터를 고치는 모든 작업

# In[13]:


# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, type):
    
    sentences_index = []
    
    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []
        
        # 디코더 입력일 경우 맨 앞에 START 태그 추가 → 1
        if type == DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])
        
        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence.split():
            # if x is not None → 'if x is not None` 가 `if x`보다는 처리 속도가 약간 빠르다고 합니다.
            # if not x is None
            # if x
            if vocabulary.get(word) is not None:
                # 사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
                # print("1. 사전에 있는 단어 → word:", word, "[vocabulary[word]]:", [vocabulary[word]], "sentence_index:", sentence_index)
            else:
                # 사전에 없는 단어면 OOV 인덱스를 추가
                sentence_index.extend([vocabulary[OOV]])
                print("2. 사전에 없는 단어 → word:", word, "[vocabulary[OOV]]:", [vocabulary[OOV]], "sentence_index:", sentence_index)
        
        # 최대 길이 검사
        if type == DECODER_TARGET:
            # 디코더 목표일 경우 맨 뒤에 END 태그 추가 → 2
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
        
        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]
        # 각 문장의 길이를 맞추기 위해 패딩을 추가한다. mex_len을 따로 명시하지 않으면 자동으로 인풋값 중 최대길이에 맞춰진다.
        # padding = "post" 옵션은 0이 뒤쪽에 붙도록 해준다. (0 → 앞에 붙으면 필요없는 정보를 먼저 확인하게 되므로)

        # 문장의 인덱스 배열을 추가
        sentences_index.append(sentence_index)
    
    print("sentences_index:", sentences_index)
    return np.asarray(sentences_index)


# seq2seq → 학습시 다음과 같이 총 3개의 데이터가 필요합니다.
# 
# 인코더 입력 : 12시 땡<br>
# 디코더 입력 : START 하루 가 또 가네요<br>
# 디코더 출력 : 하루 가 또 가네요 END

# In[14]:


# 인코더 입력 인덱스 변환
x_encoder = convert_text_to_index(question, word_to_index, ENCODER_INPUT)

# 첫 번째 인코더 입력 출력 (12시 땡)
x_encoder[0]


# In[15]:


# 디코더 입력 인덱스 변환
x_decoder = convert_text_to_index(answer, word_to_index, DECODER_INPUT)

# 첫 번째 디코더 입력 출력 (START 하루 가 또 가네요)
x_decoder[0]


# In[16]:


# 디코더 목표 인덱스 변환
y_decoder = convert_text_to_index(answer, word_to_index, DECODER_TARGET)

# 첫 번째 디코더 목표 출력 (하루 가 또 가네요 END)
y_decoder[0]


# In[17]:


# 원핫인코딩 초기화
one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words))) # 3D 배열(랭크=3)

# 디코더 목표를 원핫인코딩으로 변환
# 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
for i, sequence in enumerate(y_decoder):
    # print("i:",i, "sequence:",sequence)
    for j, index in enumerate(sequence):
        # print("j:",j, "index:",index)
        one_hot_data[i, j, index] = 1
        # print("one_hot_data[i, j, index]:", one_hot_data[i, j, index])

# 디코더 목표 설정
y_decoder = one_hot_data

# 첫 번째 디코더 목표 출력
y_decoder[0]


# # 원-핫 인코딩
# 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고,<br>다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다. 이렇게 표현된 벡터를 원-핫 벡터라고 합니다.<br>
# 이러한 표현 방식은 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다는 단점이 있습니다.
# 
# sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"<br>
# {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
# 
#     [[0. 0. 1. 0. 0. 0. 0. 0.]  #인덱스 2의 원-핫 벡터
#      [0. 0. 0. 0. 0. 1. 0. 0.]  #인덱스 5의 원-핫 벡터
#      [0. 1. 0. 0. 0. 0. 0. 0.]  #인덱스 1의 원-핫 벡터
#      [0. 0. 0. 0. 0. 0. 1. 0.]  #인덱스 6의 원-핫 벡터
#      [0. 0. 0. 1. 0. 0. 0. 0.]  #인덱스 3의 원-핫 벡터
#      [0. 0. 0. 0. 0. 0. 0. 1.]] #인덱스 7의 원-핫 벡터
# 
# softmax는 두 가지 역할을 수행한다.
# 1. 입력을 sigmoid와 마찬가지로 0과 1 사이의 값으로 변환한다.
# 2. 변환된 결과에 대한 합계가 1이 되도록 만들어 준다.
# 
# one-hot encoding은 softmax로 구한 값 중에서 가장 큰 값을 1로, 나머지를 0으로 만든다. 어떤 것을 선택할지를 확실하게 정리해 준다. one-hot encoding은 설명한 것처럼 매우 간단하기 때문에 직접 구현할 수도 있지만, 텐서플로우에서는 argmax 함수라는 이름으로 제공하고 있다.
# 
# 출처: https://pythonkim.tistory.com/20 [파이쿵]
# 
# # 결론
# 인코더 입력과 디코더 입력은 임베딩 레이어에 들어가는 인덱스 배열입니다.<br>
# 반면에 디코더 출력은 원핫인코딩 형식이 되어야 합니다.<br>
# 디코더의 마지막 Dense 레이어에서 softmax로 나오기 때문입니다.

# ### 모델 생성

# In[18]:


#--------------------------------------------
# 훈련 모델 인코더 정의
#--------------------------------------------

# 입력 문장의 인덱스 시퀀스를 입력으로 받음 → Tensor("input_1:0", shape=(None, None), dtype=float32)
encoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
# 입력 변수 당 하나의 임베딩 레이어를 생성
encoder_outputs = layers.Embedding(len(words), embedding_dim)(encoder_inputs) # 가장 간단한 형태의 임베딩은 단어의 빈도를 그대로 벡터로 사용하는 것
# Embedding() : Embedding()은 단어를 밀집 벡터로 만드는 역할을 합니다. 인공 신경망 용어로는 임베딩 층(embedding layer)을 만드는 역할을 합니다.
# Embedding()은 정수 인코딩이 된 단어들을 입력을 받아서 임베딩을 수행합니다.

# return_state가 True면 상태값 리턴
# LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim,
                                                dropout=0.1,
                                                recurrent_dropout=0.5,
                                                return_state=True)(encoder_outputs)
# → Tensor("lstm/strided_slice_3:0", shape=(None, 128), dtype=float32)

# 히든 상태와 셀 상태를 하나로 묶음
encoder_states = [state_h, state_c] # → [<tf.Tensor 'lstm/while:4' shape=(None, 128) dtype=float32>, <tf.Tensor 'lstm/while:5' shape=(None, 128) dtype=float32>]

#--------------------------------------------
# 훈련 모델 디코더 정의
#--------------------------------------------

# 목표 문장의 인덱스 시퀀스를 입력으로 받음
decoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
# decoder_outputs = layers.Embedding(len(words), embedding_dim)(decoder_inputs)
decoder_embedding = layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs)

# 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
# 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
decoder_lstm = layers.LSTM(lstm_hidden_dim,
                           dropout=0.1,
                           recurrent_dropout=0.5,
                           return_state=True,
                           return_sequences=True)

# initial_state를 인코더의 상태로 초기화 → inital_state는 RNN의 초기 상태를 지정하는 인수입니다. 초기 상태로 incoder를 decoder로 전달하는 데 사용합니다.
decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                     initial_state=encoder_states)

# 단어의 개수만큼 노드의 개수를 설정하여 원핫인코딩 형식으로 각 단어 인덱스를 출력
decoder_dense = layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#--------------------------------------------
# 훈련 모델 정의
#--------------------------------------------

# 입력과 출력으로 함수형 API 모델 생성
# 'encoder_input_data`와 `decoder_input_data`를 `decoder_target_data`로 반환하도록 모델을 정의
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 학습 방법 설정
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', # → 출력값이 one-hot encoding 된 결과로 나오고 실측 결과와의 비교시에도 실측 결과는 one-hot encoding 형태로 구성된다.
              metrics=['accuracy'])

model.summary()


# https://wikidocs.net/78127 → 모델 생성 → 참조
# 
# ㄱ. Embedding()은 (number of samples, input_length)인 2D 정수 텐서를 입력받습니다.<br>
# 이 때 각 sample은 정수 인코딩이 된 결과로, 정수의 시퀀스입니다. Embedding()은 워드 임베딩 작업을 수행하고 (number of samples, input_length, embedding word dimensionality)인 3D 텐서를 리턴합니다.
# 
# 첫번째 인자 = 단어 집합의 크기. 즉, 총 단어의 개수<br>
# 두번째 인자 = 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기<br>
# input_length = 입력 시퀀스의 길이
# 
# ㄴ. RNN → 장기 의존성 문제(the problem of Long-Term Dependencies) → LSTM
# 요약하면 LSTM은 은닉 상태(hidden state)를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며 셀 상태(cell state)라는 값을 추가하였습니다.
# LSTM은 RNN과 비교하여 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보입니다.
# 
# ㄷ. https://keras.io/ko/layers/recurrent/ → 매개변수 설명
# 
# ㄹ. https://tykimos.github.io/2018/09/14/ten-minute_introduction_to_sequence-to-sequence_learning_in_Keras/ → 케라스를 이용해 seq2seq를 10분안에 알려주기
# 
# ㅁ. 다르게 보이지만 동일한 표기
# 케라스의 functional API가 익숙하지 않은 상태에서 functional API를 사용한 코드를 보다가 혼동할 수 있는 점이 한 가지 있습니다. 바로 동일한 의미를 가지지만, 하나의 줄로 표현할 수 있는 코드를 두 개의 줄로 표현한 경우입니다.
# 
# encoder = Dense(128)(input)<br>
# 위 코드는 아래와 같이 두 개의 줄로 표현할 수 있습니다.
# 
# encoder = Dense(128)<br>
# encoder(input)

# In[19]:


#--------------------------------------------
#  예측 모델 인코더 정의
#--------------------------------------------

# 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
encoder_model = models.Model(encoder_inputs, encoder_states)

encoder_model.summary()
#--------------------------------------------
# 예측 모델 디코더 정의
#--------------------------------------------

# 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
# 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
# 따라서 타겟 데이터를 하나씩 만들어서 다음 문자들을 하나씩 예측하는 형태로 번역된 텍스트를 얻는 방법을 사용
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,)) # → Tensor("input_3:0", shape=(None, 128), dtype=float32)
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,)) # → Tensor("input_4:0", shape=(None, 128), dtype=float32)
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 임베딩 레이어
# → decoder_embedding = layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs) # Tensor("embedding_1/embedding_lookup_1/Identity_1:0", shape=(None, None, 100), dtype=float32)

# LSTM 레이어 → Tensor("lstm_1/transpose_3:0", shape=(None, None, 128), dtype=float32)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                 initial_state=decoder_states_inputs)

# 히든 상태와 셀 상태를 하나로 묶음
# [<tf.Tensor 'lstm_1/while_1:4' shape=(None, 128) dtype=float32>, <tf.Tensor 'lstm_1/while_1:5' shape=(None, 128) dtype=float32>]
decoder_states = [state_h, state_c]

# Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
decoder_outputs = decoder_dense(decoder_outputs) # Tensor("dense/truediv_1:0", shape=(None, None, len(words)), dtype=float32)

# 예측 모델 디코더 설정
decoder_model = models.Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

decoder_model.summary()


# 예측 모델은 이미 학습된 훈련 모델의 레이어들을 그대로 재사용합니다. 예측 모델 인코더는 훈련 모델 인코더과 동일합니다. 그러나 예측 모델 디코더는 매번 LSTM 상태값을 입력으로 받습니다. 또한 디코더의 LSTM 상태를 출력값과 같이 내보내서, 다음 번 입력에 넣습니다. 
# 
# 이렇게 하는 이유는 LSTM을 딱 한번의 타임 스텝만 실행하기 때문입니다. 그래서 매번 상태값을 새로 초기화 해야 합니다. 이와 반대로 훈련할때는 문장 전체를 계속 LSTM으로 돌리기 때문에 자동으로 상태값이 전달됩니다. 

# ### 훈련 및 테스트

# In[20]:


# 인덱스를 문장으로 변환
def convert_index_to_text(indexs, vocabulary): 
    
    sentence = ''
    
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX: # → 2
            # 종료 인덱스면 중지
            break;
        if vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 OOV 단어를 추가
            sentence.extend([vocabulary[OOV_INDEX]])
            
        # 빈칸 추가
        sentence += ' '

    return sentence


# In[21]:


# 에폭 반복
# for epoch in range(20):
for epoch in range(5):
    print('Total Epoch :', epoch + 1)

    # 훈련 시작
    history = model.fit([x_encoder, x_decoder],
                        y_decoder,
                        epochs=100,
                        batch_size=64,
                        verbose=0)
                        # verbose=1)
    
    # 모델 저장
    model.save('my_model.h5')
    
    # 정확도와 손실 출력
    print('accuracy :', history.history['accuracy'][-1])
    print('loss :', history.history['loss'][-1])
    
    # 문장 예측 테스트
    # (1 지망 학교 떨어졌어) -> (위로 해 드립니다)
    # → indexs = [53 27 34  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    input_encoder = x_encoder[1].reshape(1, x_encoder[1].shape[0]) # 2D 배열(랭크=2)
    input_decoder = x_decoder[1].reshape(1, x_decoder[1].shape[0])
    # (3 박 4일 놀러 가고 싶다) -> (여행 은 언제나 좋죠)
    # input_encoder = x_encoder[2].reshape(1, x_encoder[2].shape[0])
    # input_decoder = x_decoder[2].reshape(1, x_decoder[2].shape[0])

    results = model.predict([input_encoder, input_decoder]) # 3D 배열(랭크=3) → (1, 30, 70)

    # 결과의 원핫인코딩 형식을 인덱스로 변환
    # 1축을 기준으로 가장 높은 값의 위치를 구함
    indexs = np.argmax(results[0], 1) # results[0] → (30, 70)
    # np.argmax(x)는 x 배열을 1차원으로 평면화된 배열에 대해 최대값을 갖는 인덱스를 반환하고,
    # np.argmax(x, axis=0)는 첫번째 축인 row 방향(세로방향)으로 구성되는 요소 중 최대인 인덱스를 반환하며,
    # y1 = np.argmax(x, axis=1)는 두번째 축인 column 방향(가로방향)으로 구성되는 요소 중 최대인 인덱스를 반환합니다.
    
    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
    print(sentence)
    print()
   


# ㄱ. 학습이 진행될수록 예측 문장이 제대로 생성되는 것을 볼 수 있습니다. 다만 여기서의 예측은 단순히 테스트를 위한 것이라, 인코더 입력과 디코더 입력 데이터가 동시에 사용됩니다. 아래 문장 생성에서는 예측 모델을 적용하기 때문에, 오직 인코더 입력 데이터만 집어 넣습니다.<br>
# 
# ㄴ. verbose: Integer. 0, 1, or 2.<br>
# Verbosity mode. 
# 
# 0 = silent,<br> 
# 1 = progress bar,<br> 
# 2 = one line per epoch.
# 
# ㄷ. 자료<br>
# Q: 12시 땡<br>
# A: 하루 가 또 가네요
# 
# Q: 1 지망 학교 떨어졌어<br>
# A: 위로 해 드립니다
# 
# Q: 3 박 4일 놀러 가고 싶다<br>
# A: 여행 은 언제나 좋죠

# ### 문장 생성

# In[22]:


# 예측을 위한 입력 생성
def make_predict_input(sentence):

    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)
    
    return input_seq


# In[23]:


# 텍스트 생성 → 타겟 데이터를 하나씩 만들어서 다음 문자들을 하나씩 예측하는 형태로
def generate_text(input_seq):
    
    # 입력을 인코더에 넣어 마지막 상태 구함 → 예측 모델 인코더
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX # → 1
    # target_seq[0][0] = STA_INDEX # → 1
    
    # 인덱스 초기화
    indexs = []
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
        
    return sentence


# 제일 첫 단어는 START로 시작합니다. 그리고 출력으로 나온 인덱스를 디코더 입력으로 넣고 다시 예측을 반복합니다. 상태값을 받아 다시 입력으로 같이 넣는 것에 주의하시기 바랍니다. END 태그가 나오면 문장 생성을 종료합니다.
# 
# ㄱ. 인코더 모델과 디코더 모델을 따로 구분하여 다시 정의하는 이유는 인코더로부터 얻은 상태를 decoder의 초기 상태로 한 다음, 처음 시작 문자\t로 시작하여 디코더가 예측한 문자로 타겟 입력을 계속 바꾸어 다음 문자를 예측할 수 있게 하기 위함이다.

# In[24]:


# 문장을 인덱스로 변환
input_seq = make_predict_input('3박4일 놀러가고 싶다')
input_seq


# In[25]:


# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
sentence


# 데이터셋에 있는 문장과 똑같은 입력을 넣으니, 역시 정확히 일치하는 답변이 출력되었습니다.

# In[26]:


# 문장을 인덱스로 변환
input_seq = make_predict_input('3박4일 같이 놀러가고 싶다')
input_seq


# In[27]:


# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
sentence


# 데이터셋 문장에서는 없던 '같이'라는 단어를 추가해 보았습니다. 그래도 비슷한 의미란 것을 파악하여 동일한 답변이 나왔습니다.

# In[28]:


# 문장을 인덱스로 변환
# input_seq = make_predict_input('3박4일 같이 놀러가려고')
input_seq = make_predict_input('같이 놀러가려고')
input_seq


# In[29]:


# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
sentence


# 하지만 데이터셋에 없던 '가려고'로 입력을 수정하니, 전혀 다른 문장이 출력되었습니다. 이는 우리가 데이터의 일부인 100개 문장만 학습했기 때문입니다. 데이터의 개수를 늘려서 훈련할수록 일반화 능력이 더욱 높아집니다.
