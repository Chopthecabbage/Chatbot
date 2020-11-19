import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chopthecabbagge-964d5c83271f.json"

import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import json
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('OriginalChatbotDataset.json') as file:
   data = json.load(file)

from sklearn.preprocessing import LabelEncoder

sentences = []  # 패턴 문장들
labels = []  # 패턴 라벨들
classes = []  # 태그 종류들
responses = []  # 응답 문장들

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

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels) # → 피팅하고 라벨숫자로 변환한다.

vocab_size = 10000

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 20
trunc_type = 'post'

# 태그 단어
OOV = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=OOV) # adding out of vocabulary token
tokenizer.fit_on_texts(sentences) # 문자 데이터를 입력받아서 리스트의 형태로 변환합니다.
word_index = tokenizer.word_index # 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환합니다.
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, truncating=trunc_type, maxlen=max_sequences)

model = keras.models.load_model("형스비")

ON = 1
OFF = 0
voice = OFF

def select():
    global voice
    if r.get() == ON:
        voice = ON
    elif r.get() == OFF:
        voice = OFF

def blurb():
    if len(e.get()) < 1:
        messagebox.showinfo("도움말", "입력을 완성해주세요")
        return

    chatWindow.config(state=NORMAL)

    me = "나: " + e.get()
    chatWindow.insert(END, me + "\n")

    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([e.get()]),
                                         truncating=trunc_type, maxlen=max_sequences))
    category = encoder.inverse_transform([np.argmax(result)])  # 인덱스(라벨숫자)를 입력하면 본래 값을 반환

    global voice
    for i in data['intents']:
        if i['tag'] == category:
            compare = category.astype('int')
            if compare < 1000:
                chatWindow.insert(END, "봇: " + str(i['responses']) + "\n")
                check = gTTS.isEnglishOrKorean(e.get())
                if check == "k" and voice == ON:
                    gTTS.run_quickstart(str(i['responses']), 1)
                elif check == "e" and voice == ON:
                    gTTS.run_quickstart(str(i['responses']), 2)
            else:
                chatWindow.insert(END, "봇: " + str(np.random.choice(i['responses'])) + "\n")
                check = gTTS.isEnglishOrKorean(e.get())
                if check == "k" and voice == ON:
                    gTTS.run_quickstart(str(np.random.choice(i['responses'])), 1)
                elif check == "e" and voice == ON:
                    gTTS.run_quickstart(str(np.random.choice(i['responses'])), 2)

    e.delete(0, END)  # 삭제
    e.focus_set()  # 포커스 설정

    chatWindow.config(state=DISABLED)

def pressed(event):
    if len(e.get()) < 1:
        messagebox.showinfo("도움말", "입력을 완성해주세요")
        return

    chatWindow.config(state=NORMAL)

    me = "나: " + e.get()
    chatWindow.insert(END, me + "\n")

    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([e.get()]),
                                         truncating=trunc_type, maxlen=max_sequences))
    category = encoder.inverse_transform([np.argmax(result)])  # 인덱스(라벨숫자)를 입력하면 본래 값을 반환
    global voice
    for i in data['intents']:
        if i['tag'] == category:
            compare = category.astype('int')
            if compare < 1000:
                chatWindow.insert(END, "봇: " + str(i['responses']) + "\n")
                check = gTTS.isEnglishOrKorean(e.get())
                if check == "k" and voice == ON:
                    gTTS.run_quickstart(str(i['responses']), 1)
                elif check == "e" and voice == ON:
                    gTTS.run_quickstart(str(i['responses']), 2)
            else:
                chatWindow.insert(END, "봇: " + str(np.random.choice(i['responses'])) + "\n")
                check = gTTS.isEnglishOrKorean(e.get())
                if check == "k" and voice == ON:
                    gTTS.run_quickstart(str(np.random.choice(i['responses'])), 1)
                elif check == "e" and voice == ON:
                    gTTS.run_quickstart(str(np.random.choice(i['responses'])), 2)

    e.delete(0, END)  # 삭제
    e.focus_set()  # 포커스 설정

    chatWindow.config(state=DISABLED)

from tkinter import messagebox
from tkinter import scrolledtext
import tkinter.font
import gTTS
from tkinter.filedialog import *
# import speech_recognition as sr
'''
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = " "

        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))

    return said
'''
color = '#%02x%02x%02x' % (64, 204, 208) # '#40E0D0'  # or use hex if you prefer

root = Tk()
root.title('형스비(Hyungxbi)')
root.geometry('400x520+760+300')
root.configure(bg=color)
root.iconbitmap(r'Ver.ico')
root.resizable(0, 0) # 창의 x와 y 의 크기 조절 불가능

# 폰트
font = tkinter.font.Font(family="맑은 고딕", size=12, slant="italic")

def clear():
    chatWindow.config(state=NORMAL)
    chatWindow.delete("1.0", "end")
    chatWindow.config(state=DISABLED)

def save():
    f = asksaveasfile(mode="w", defaultextension=".txt")
    if f is None:
        return
    ts = str(chatWindow.get(1.0, END))
    f.write(ts)
    f.close()

def f12():
    messagebox.showinfo("챗봇 정보", "버전: 2020/11/14 → 형스비(Hyungxbi)_ver1\n" + "채팅: 엔터, 전송 버튼을 통해 대화 내용 전달")

# 메인 메뉴바
main_menu = Menu(root) # 윈도우에 메뉴바 추가

# 드롭다운 메뉴바
file_menu = Menu(main_menu, tearoff=0) # 상위 메뉴 탭 항목 추가
file_menu.add_command(label='초기화', command=clear)
file_menu.add_command(label='저장', command=save)
file_menu.add_separator()
file_menu.add_command(label='종료', command=root.destroy)
main_menu.add_cascade(label='파일', menu=file_menu) # 상위 메뉴 탭 설정

help = Menu(main_menu, tearoff=0)
help.add_command(label="챗봇 정보", command=f12)
main_menu.add_cascade(label='도움말', menu=help) # 항목 추가

root.config(menu=main_menu)

# 메시지 창
chatWindow = scrolledtext.ScrolledText(root, bd=1, bg='white', width=50, height=8, font=font)
# borderwidth=5, highlightthickness=1, bg='#15202b', fg='#16202A',
chatWindow.configure(state='disabled') # 텍스트 위젯을 읽기 전용으로 설정
chatWindow.place(x=5, y=5, width=390, height=385)

# 텍스트 창
e = Entry(root, bd=5, font=font)
e.bind("<Return>", pressed)
e.place(x=5, y=400, width=390, height=50)

# 전송 버튼
Button = Button(root, text='전송', command=blurb, bd=5, bg='yellow', fg="red", activebackground='light blue', width=10, height=5, font=('맑은 고딕', 20))
Button.place(x=300, y=450, width=100, height=50)

# 라디오 버튼
r = IntVar()
r1 = Radiobutton(root, text='음성 켜기', variable=r, value=ON, command=select, indicatoron=0, bd=3, bg='pink', fg="purple", font=font)
r1.place(x=0, y=450, width=100, height=50)
r2 = Radiobutton(root, text='음성 끄기', variable=r, value=OFF, command=select, indicatoron=0, bd=3, bg='pink', fg="purple", font=font)
r2.place(x=100, y=450, width=100, height=50)

root.mainloop()