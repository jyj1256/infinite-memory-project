import os
from openai import OpenAI
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import pinecone
import datetime



def gpt3_embedding(content, model='text-embedding-ada-002'):
    openai_api_key = "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"
    client = OpenAI(api_key=openai_api_key)
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = client.embeddings.create(input=content, model=model)
    print(response)
    vector = response['data'][0]['embedding']  # this is a normal list
    print(vector)
    return vector

def get_embedding(text, model="text-embedding-ada-002"):
    openai_api_key = "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"
    client = OpenAI(api_key=openai_api_key)
    text = text.replace("\n", " ")
    print(text)
    return print(client.embeddings.create(input = [text], model=model).data[0].embedding)

content ="안녕"
get_embedding(content, model='text-embedding-ada-002')

