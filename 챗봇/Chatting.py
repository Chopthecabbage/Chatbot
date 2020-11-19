import json
# from collections import OrderedDict
# data = OrderedDict()

file_path = "./OriginalChatbotDataset.json"

greeting_patterns = ["반가워", "반갑습니다", "안녕", "안녕하세요", "잘 지냈어", "잘 지냈어요", "잘지내요"]
greeting_responses = ["나도 반가워.", "어서오세요... 반갑습니다.", "너도 안녕...", "잘지내고 있어...", "만나서 반갑습니다..."]

goodbye_patterns = ["잘가 담에 보자", "안녕히 가세요", "다음에 다시보자", "다음에 다시봬요"]
goodbye_responses = ["나중에 보자.", "좋은밤 되세요.", "조심히 들어가세요.", "내일 다시봬요.", "너도 잘가."]

thanks_patterns = ["고마워", "고맙습니다", "고맙지만", "도움이 많이 되었어요", "도와줘서 감사합니다", "정말 도움이 많이 됐어", "감사합니다", "쓸만하네"]
thanks_responses = ["언제든지 도와줄게.", "나야말로 고맙지.", "그래 고생했어.", "내가 널 도와줘서 나도 기쁘다.", "별거 아냐... 도움이 되서 나도 좋다."]

noanswer_patterns = ["바보", "씨발", "개새끼", "나쁜놈"]
noanswer_responses = ["이해를 하지 못했습니다. 정보를 더 주세요...", "질문의 요지가 있습니까?", "어떤 대답을 원하시는 건가요?", "죄송합니다. 이해를 하지 못했습니다.", "똑바로 입력해라..."]

lol1_patterns = ["내 롤 티어는", "내 티어는", "롤 티어"]
lol1_responses = ["플레티어4", "플4...신지드 장인입니다...", "플레티넘입니다..."]

lol2_patterns = ["탑 라인은", "탑을 하는 사람들은", "탑 라이너는"]
lol2_responses = ["정신병자입니다.", "탑신병자입니다..."]

lol3_patterns = ["탑이 죽으면", "탑이 말리면", "탑 차이는"]
lol3_responses = ["정글탓입니다.", "정글차이..."]

lol4_patterns = ["신지드는", "신지드는 어떤 챔피언이야", "신지드는 좋아"]
lol4_responses = ["주인님이 쓰신다면 최고의 챔피언입니다..."]

greeting_eng_patterns = ["Hi there", "How are you", "Is anyone there?", "Hello", "Good day"]
greeting_eng_responses = ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"]

goodbye_eng_patterns = ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"]
goodbye_eng_responses = ["See you!", "Have a nice day", "Bye! Come back again soon."]

thanks_eng_patterns = ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"]
thanks_eng_responses = ["Happy to help!", "Any time!", "My pleasure"]

options_patterns = ["How you could help me?", "What you can do?", "What help you provide?", "How you can be helpful?", "What support is offered"]
options_responses = ["I can guide you through Adverse drug reaction list, Blood pressure tracking, Hospitals and Pharmacies", "Offering support for Adverse drug reaction, Blood pressure, Hospitals and Pharmacies"]

pharmacy_search_patterns = ["Find me a pharmacy", "Find pharmacy", "List of pharmacies nearby", "Locate pharmacy", "Search pharmacy"]
pharmacy_search_responses = ["Please provide pharmacy name"]
pharmacy_context = ["search_pharmacy_by_name"]

hospital_search_patterns = ["Lookup for hospital", "Searching for hospital to transfer patient", "I want to search hospital data", "Hospital lookup for patient", "Looking up hospital details"]
hospital_search_responses = ["Please provide hospital name or location"]
hospital_search_context = ["search_hospital_by_params"]

import pandas as pd
# 챗봇 데이터 로드
chatbot_data = pd.read_csv('./ChatbotData.csv', encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])

# 데이터의 일부만 학습에 사용
question = question[:1000]
answer = answer[:1000]

data = { }
data['intents'] = []

for x in range(len(question)):
    data['intents'].append({
        "tag": str(x),
        "patterns": question[x],
        "responses": answer[x],
        "context": ""
    })
    print("태그: " + str(x))
    print("패턴: " + question[x])
    print("responses: " + answer[x])

'''
import random # 랜덤...

list = []
ran_num = random.randint(0, 11823)
for i in range(1000):
    while ran_num in list:
        ran_num = random.randint(0, 11823)
    list.append(ran_num)
    data['intents'].append({
        "tag": str(i),
        "patterns": question[ran_num],
        "responses": answer[ran_num],
        "context": ""
    })
    print("태그: " + str(i))
    print("패턴: " + question[ran_num])
    print("responses: " + answer[ran_num])
'''

with open(file_path, 'w') as outfile:
     json.dump(data, outfile)

print("데이터셋:", len(data['intents']))

json_data = {}
json_data['intents'] = []
with open(file_path, "r") as json_file:
     json_data = json.load(json_file)

json_data['intents'].append({
    "tag": "1000",
    "patterns": greeting_patterns,
    "responses": greeting_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1001",
    "patterns": goodbye_patterns,
    "responses": goodbye_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1002",
    "patterns": thanks_patterns,
    "responses": thanks_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1003",
    "patterns": noanswer_patterns,
    "responses": noanswer_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1004",
    "patterns": lol1_patterns,
    "responses": lol1_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1005",
    "patterns": lol2_patterns,
    "responses": lol2_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1006",
    "patterns": lol3_patterns,
    "responses": lol3_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1007",
    "patterns": lol4_patterns,
    "responses": lol4_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1008",
    "patterns": greeting_eng_patterns,
    "responses": greeting_eng_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1009",
    "patterns": goodbye_eng_patterns,
    "responses": goodbye_eng_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1010",
    "patterns": thanks_eng_patterns,
    "responses": thanks_eng_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1011",
    "patterns": options_patterns,
    "responses": options_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1012",
    "patterns": pharmacy_search_patterns,
    "responses": pharmacy_search_responses,
    "context": ""
})
json_data['intents'].append({
    "tag": "1013",
    "patterns": hospital_search_patterns,
    "responses": hospital_search_responses,
    "context": ""
})

with open(file_path, 'w') as outfile:
    json.dump(json_data, outfile, indent=4)

print("데이터셋:", len(json_data['intents']))