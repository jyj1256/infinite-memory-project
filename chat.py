import os
from openai import OpenAI
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone
from flask import Flask, render_template, request

'''
uuid 모듈은 Universally Unique IDentifier(범용 고유 식별자)를 생성하는데 사용됩니다.
uuid4 함수는 랜덤한 UUID를 생성합니다. UUID는 전 세계적으로 고유한 값을 생성하는 데 사용되며, 
주로 데이터베이스 레코드나 파일 등을 식별하는 데 활용됩니다.
'''

#파일 경로를 입력으로 받아 해당 파일을 UTF-8 인코딩으로 읽어들인 후, 그 내용을 문자열로 반환
def open_file(filepath):   
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# 파일 저장하는 함수
def save_file(filepath, content):  
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

#제이슨 파일 불러오는 함수
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

# 제이슨파일 저장하는함수
def save_json(filepath, payload):                                                   #payload: JSON 형식으로 저장하고자 하는 데이터를 담고 있는 변수
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)  #json.dump() 함수는 Python에서 JSON 데이터를 파일로 저장하는 데 사용

#시간,날짜체계 만들기
def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

#문장 벡터로 임베딩
def gpt3_embedding(content, model='text-embedding-ada-002'):
    client = OpenAI(api_key=openai_api_key)
    # content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    # response = client.embeddings.create(input=content, model=model)
    # vector = response['data'][0]['embedding']  # this is a normal list
    content = content.replace("\n", " ")
    vector = client.embeddings.create(input = [content], model=model).data[0].embedding
    return vector



#gpt-3.5-turbo-instruct
#gpt-3.5-turbo-0125
#ft:gpt-3.5-turbo-0613:personal::8x76h0Py
#gpt api 이용해서 파인튜닝된 모델 사용
def gpt3_completion(prompt, model='ft:gpt-3.5-turbo-0613:personal::8x76h0Py'):
    max_retry = 5
    retry = 0
    # prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = client.chat.completions.create(
                model = model,
                messages=[
                    {"role": "system", "content": "날짜 체계를 고려해서 두번째 문장에대한 대답해"},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

# 유사도로 가까운거 뽑은것들 텍스트로 쭉
def load_conversation(results):    #results = vdb.query(vector=vector, top_k=convo_length)
    result = list()   #-> result라는 리스트 만듬
    for m in results['matches']:   #results에는 matches라는게있음 그만큼 반복 -> 아마 top_k 일듯?
        info = load_json('nexus/%s.json' % m['id'])   
        result.append(info)                           #result 리스트에 info 삽입
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()

'''
parsed_data = json.loads(data)

# pinecone 문서보니까 벡터 똑같은값이 스코어가 작아서 일단은 스코어 작은거 id 찾는걸로함

lowest_score = float("inf")  가장 작은 값을 찾기 위해 초기값을 설정
lowest_score_id = None
for m in parsed_data["matches"]:
    if m["score"] < lowest_score:
        lowest_score = m["score"]
        lowest_score_id = m["id"]

print("best usado:", lowest_score_id)
'''
def save_list_file_for_finetuning(file_name, content):     #-> 파인튜닝할용도로 prompt와 output저장
    try:
        # 파일을 쓰기 모드로 엽니다.
        with open(file_name, 'w') as file:
            # content를 파일에 씁니다.
            file.write(content)
        print(f'파일 "{file_name}"에 성공적으로 저장되었습니다.')
    except Exception as e:
        print(f'파일 "{file_name}" 저장 중 오류가 발생했습니다: {e}')


# 유사도로 가까운거 뽑은것들 텍스트로 쭉
def load_conversation2(results):    #results = vdb.query(vector=vector, top_k=convo_length)
    result = list()   #-> result라는 리스트 만듬
    #parsed_data = json.loads(results)           #첫번째 문제  the JSON object must be str, bytes or bytearray, not QueryResponse
    highest_score = float("-inf")  # 가장 작은 값으로 초기화
    highest_score_id = None
    for m in results["matches"]:
        if m["score"] > highest_score:
            highest_score = m["score"]
            highest_score_id = m["id"]   #lowest_score_id에 제일 유사한 문장의 벡터의 id가 들어가있음  여기까진 잘뽑아짐
    print(highest_score_id)
    info = load_json('nexus/%s.json' % highest_score_id)  #두번째 에러: [Errno 2] No such file or directory: 'nexus/3eba54a1-95eb-4996-b0e9-f2e4f04d973c.json'
    result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip() 


#미리 만들 변수  ->a : 서버에 꺼내와야함, 



if __name__ == '__main__':
    convo_length = 5 #유사도 가장 높은 거 3개 뽑는 용도
    openai_api_key = open_file('key_openai.txt')
    client = OpenAI(api_key=openai_api_key)
    pinecone.init(api_key=open_file('key_pinecone.txt'), environment='gcp-starter')
    # pinecone.create_index("raven-mvp", dimension=1586, metric="cosine") #생성
    # print(pinecone.describe_index("raven-mvp")) #설명
    vdb = pinecone.Index("raven-mvp")
    while True:
        now = datetime.datetime.now()                      #현재시간 가져오기  #now에는 현재시간 들어감
        formatted_date_time = now.strftime("%Y%m%d-%H:%M")  #날짜체계가들어감 ex) -> 20240307-21:50
        hangletime = str(formatted_date_time)             #ormatted_date_time과 똑같은게 들어감
        print(type(formatted_date_time))
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()                                            #JSON 형식으로 저장하고자 하는 데이터를 담고 있는 변수를 리스트로 형변환
        a = input('\n\nUSER: ')                  # 사용자가 프로그램에게 말하고싶은것,질문하고싶은건 ex) -> 내가 4월27일날 뭐하기로 했더라
        timestamp = time()          
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('USER', timestring, a)
        message = hangletime + " : " + a        # ex)-> 20240307-21:50 : 내가 4월 27일날 뭐하기로 했지?
        print(message)                          #사용자가 말한것 출력
        vector = gpt3_embedding(message)        #사용자가 말한것 벡터로 임베딩
        unique_id = str(uuid4())      #uid를 생성 
        metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id} #메타데이터 만들고
        save_json('nexus/%s.json' % unique_id, metadata) #제이슨파일만들어서 저장
        #payload.append((unique_id, vector))
        #################################여기까지가 사용자의 답변 #################



        #### search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=convo_length) # 유사문장 찾기(vector값, top_k= 유사문장 갯수)
        print(type(results))   #<class 'pinecone.core.client.model.query_response.QueryResponse'>
        print(results)
        '''
        {'matches': [{'id': '3eba54a1-95eb-4996-b0e9-f2e4f04d973c',
              'score': 0.786891937,
              'values': []}],
 'namespace': '',
 'usage': {'readUnits': 5}}
        '''
        #'3eba54a1-95eb-4996-b0e9-f2e4f04d973c',  내 4월5일날 누구하고 가평가기로 했지?
        #3eba54a1-95eb-4996-b0e9-f2e4f04d973c     내가 4월27일날 뭐하기로 했지?
        #3eba54a1-95eb-4996-b0e9-f2e4f04d973c  안녕
      
        #내가 예상하기론 그때 파인콘 벡터 한번 지워서 파인콘에 3eba54a1-95eb-4996-b0e9-f2e4f04d973c 이거 하나밖에 안들어있어서 그런듯
        conversation = load_conversation2(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id' # 결과는 'id'가 포함된 'matches'가 포함된 DICT여야 합니다
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)  #사용자가 현재말한 문장과 conversation을 합치기 ex) 20240226-13:56 : 오 이 자주색 코트 이쁘다 내일 인터넷으로 찾아봐야겠다.  20240227-16:17 : 어제 내가 이쁘다고한 코트 무슨색이었지?
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)    #gptapi에 prompt 넣고 파인튜닝된 gpt의 답변 구하기   ex) 어제 자주색 코트가 이쁘다며 말씀하셨어요.
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('RAVEN', timestring, output)
        message = output      #ex)message에는 어제 자주색 코트가 이쁘다며 말씀하셨어요. 가 들어가있음
        

        #vector = gpt3_embedding(message)      # ex)message에는 어제 자주색 코트가 이쁘다며 말씀하셨어요. 가 임베딩되어있음
        #답변은 굳이 임베딩할필요가 없을수도?  답변은 그냥 사용자에게 보여주기만해도 될듯 -> vdb에 넣을게 아니라서 임베딩 안해도될듯

        #############################################안해도되는것###############################
        '''unique_id = str(uuid4())    #uid 생성
        metadata = {'speaker': '나동반', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id} #메타데이터 만들기
        save_json('nexus/%s.json' % unique_id, metadata)   #답변을 json파일 만들어서 저장  metadata가 payload
        payload.append((unique_id, vector))
        vdb.upsert(payload)  #payload는 리스트임'''
        #############################################안해도되는것###############################
        
        #vdb.upsert(vector)
        payload.append((unique_id, vector))
        vdb.upsert(payload)
        #vdb.upsert(vectors=[{"id":unique_id,"values":vector}])

        print('\n\n나동반: %s' % output) # output 출력 ex)어제 자주색 코트가 이쁘다며 말씀하셨어요.  ->성공!!

        file_name = "listup_file_for_finetuning"   #->그냥 텍스트 파일
        content = prompt+'\n'+output+'\n\n\n'  #-> ex) 20240226-13:56 : 오 이 자주색 코트 이쁘다 내일 인터넷으로 찾아봐야겠다.  20240227-16:17 : 어제 내가 이쁘다고한 코트 무슨색이었지? \n 어제 자주색 코트가 이쁘다며 말씀하셨어요. \n\n\n
        save_list_file_for_finetuning(file_name, content)

        #vdb와 nexus에 들어갈것 a, gpt의답변은 들어가지안아도됨
        #유사도 에서는 하나만 뽑을것 ->고쳐야될것 :load_conversation()

        #확인해볼것 conversation에 뭐가들어가는지
        



        # output = gpt3_completion(prompt)  gpt 대답  
        # prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)  gpt에게 할말   





'''
USER: 내가 이쁘다고한 코트 무슨색이었지?
20240308-02:22 : 내가 이쁘다고한 코트 무슨색이었지?
<class 'pinecone.core.client.model.query_response.QueryResponse'>
{'matches': [{'id': '4dcb4b8a-e006-4199-b728-5959a6b32cee',
              'score': 0.912704527,
              'values': []},
             {'id': '3eba54a1-95eb-4996-b0e9-f2e4f04d973c',
              'score': 0.769784689,
              'values': []}],
 'namespace': '',
 'usage': {'readUnits': 5}}
4dcb4b8a-e006-4199-b728-5959a6b32cee

나동반: 당신이 이쁘다고 언급한 코트는 자주색 코트입니다. 자주색 코트는 독특한 색상으로, 당신의 스타일과 성격을 반영해줄 것 같아요.
 자주색은 여러 가지 의미와 감정을 상징하는데요, 전통적으로는 독립심, 독창성, 상상력, 영감 등을 나타냅니다. 
 또한, 세련되고 고급스러운 분위기를 연출해줄 수 있는 컬러 중 하나이기도 해요. 그러므로, 자주색 코트를 선택하면 독특하면서도 트렌디한 룩을 연출할 수 있을 거예요. 
 그런데, 자주색 코트를 찾기 위해 인터넷을 찾아보는 건 어떠세요? 서로 어울리는 아이템을 찾아볼 수도 있을 거에요. 
 어울리는 컬러, 스타일 등을 고려해서 좋은 선택을 할 수 있을 거예요.
'''

'''
USER: 친구 결혼식이 언제였더라?
20240314-14:55 : 친구 결혼식이 언제였더라?
<class 'pinecone.core.client.model.query_response.QueryResponse'>
{'matches': [{'id': 'e29e49f6-5944-4006-b50c-ffebb628f5f1',
              'score': 0.899677813,
              'values': []},
             {'id': '0a1a2a05-1d14-4a37-9d11-d45818daed8b',
              'score': 0.875287175,
              'values': []},
             {'id': 'ff93a0fc-1e44-4ca1-8de6-7a0b56ec7a9c',
              'score': 0.870127797,
              'values': []},
             {'id': 'a4306a68-1d18-4fba-a8f1-c8283e34a095',
              'score': 0.8672387,
              'values': []},
             {'id': '09a0c3e7-393f-4dfa-8443-ab1eb619eda8',
              'score': 0.863476157,
              'values': []}],
 'namespace': '',
 'usage': {'readUnits': 5}}
e29e49f6-5944-4006-b50c-ffebb628f5f1


나동반: 친구 결혼식은 내일 모레에 진행됩니다. 이전 대화에서 내일 모래에 친구의 결혼식에 참석한다고 하셨는데요. 그렇다면, 준비물이나 드레스 코드 같은 부분을 고민하고 계실지 궁금합니다. 결혼식에 어울리는 복장으로 어떤 스타일이 좋을지 고민하고 계신다면, 저희가 도움을 드릴 수 있을 것 같아요. 드레스 스타일, 색상 또는 액세서리 등에 고민이 있으신가요?
파일 "listup_file_for_finetuning"에 성공적으로 저장되었습니다.
<class 'str'>
'''

#나 내일 학교 가는날인가? -> 나 내일 무슨수업이 있더라?