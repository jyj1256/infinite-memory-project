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

#넥서스와 vdb에 임의로 데이터넣기위한 테스트 파이썬 파일

def gpt3_embedding(content, model='text-embedding-ada-002'):
    client = OpenAI(api_key='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')
    # content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    # response = client.embeddings.create(input=content, model=model)
    # vector = response['data'][0]['embedding']  # this is a normal list
    content = content.replace("\n", " ")
    vector = client.embeddings.create(input = [content], model=model).data[0].embedding
    return vector

def open_file(filepath):   
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
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



pinecone.init(api_key=open_file('key_pinecone.txt'), environment='gcp-starter')
#pinecone.init(api_key="f907c6e2-ca80-4c89-9614-be9befcad63e", environment="gcp-starter")
vdb = pinecone.Index("raven-mvp")


now = datetime.datetime.now() 
formatted_date_time = now.strftime("%Y%m%d-%H:%M")
hangletime = str(formatted_date_time)
payload = list() 
a = input('\n\nUSER: ')
timestamp = time() 
timestring = timestamp_to_datetime(timestamp)
message = hangletime + " : " + a
print(message)

unique_id = str(uuid4())
print(unique_id) 
metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
save_json('nexus/%s.json' % unique_id, metadata)

vector = gpt3_embedding(message)
payload.append((unique_id, vector))
#print(vector)
#print(type(vector)) #list
#vdb.upsert(vectors=[{"id":unique_id,"values":vector}])
vdb.upsert(payload)
print('넥서스와 vdb에 저장완료')

#나 오늘 오후 12시30분쯤에 윤회진 교수님이랑 졸업작품에대해서 이야기 하기로 했어
# 내일은 학교 안가는 날이야 너무 기분이 좋아!

'''#USER: 내일은 학교 안가는 날이야 너무 기분이 좋아!
20240314-14:47 : 내일은 학교 안가는 날이야 너무 기분이 좋아!
5921eaa4-29e7-4fee-8e9b-e831be66f1d1
넥서스와 vdb에 저장완료'''


#내일은 사회봉사라는 수업이 있어
'''
USER: 내일은 사회봉사라는 수업이 있어
20240318-18:37 : 내일은 사회봉사라는 수업이 있어
55c9e742-9f6a-420c-a519-1e88769072ae
넥서스와 vdb에 저장완료'''


#목요일까지 해야할 졸업작품 숙제가 있어
'''USER: 목요일까지 해야할 졸업작품 숙제가 있어
20240319-14:57 : 목요일까지 해야할 졸업작품 숙제가 있어
1b543015-b39d-46f6-8bc3-8aabc2b37c8c
넥서스와 vdb에 저장완료'''


'''
USER: 방금 빌리 아일리시의 Bad Guy 들었는데 좋더라
20240319-15:44 : 방금 빌리 아일리시의 Bad Guy 들었는데 좋더라
3f6f9b31-74ea-4b87-8a13-612ec29f3521
넥서스와 vdb에 저장완료
'''
#이번엔 무슨노래 들을까?

#7일후에 지옥이 강림할거야
'''
USER: 7일후에 지옥이 강림할거야
20240319-16:46 : 7일후에 지옥이 강림할거야
50c4ded1-b2a1-4fb8-9059-6c522e6b4dcf
넥서스와 vdb에 저장완료
'''
#내가 7일후에 무슨일이 일어날거라고 했지?