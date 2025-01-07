# GPT-Based Food Recommend System Development
# 부제: 개인 맞춤 알레르기 회피 식품 추천 시스템
# 내용: 개인이 보유하고 있는 알레르기 유발 물질이 함유되지 않은 식품을 추천하는 시스템으로, 알레르기 정보가 포함된 식품을 선택하지 않도록 도와줄 수 있음
# 개발자: 박보현

# 참고: https://github.com/lsjsj92/recommender_system_with_Python/blob/master/009_chatgpt_recsys.ipynb

#Streamlit 앱 코드
import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer, util
import openai
import os
import sys
from dotenv import load_dotenv
import copy
import ast
from typing import List
import re


# .env 파일에서 API 키 가져오기
#load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# SentenceTransformer 모델 로드
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# 사전 처리된 식품 데이터셋 로드
food_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Kaggle/food_data_em.csv')

# hf_embeddings 열을 리스트로 변환
#def parse_embedding(embedding_str):
#    try:
#        return np.array(ast.literal_eval(embedding_str)).astype(np.float32)
#    except:
#        return np.zeros((512,), dtype=np.float32)  # 임시로 512차원 영벡터로 반환

#food_data['hf_embeddings'] = food_data['hf_embeddings'].apply(parse_embedding)


# hf_embeddings 컬럼의 문자열에서 숫자 추출하여 리스트로 변환하는 함수
def parse_embeddings(embedding_str):
    # 문자열에서 숫자 추출
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", embedding_str)
    # 문자열을 float로 변환
    numbers = [float(num) for num in numbers]
    # 512차원으로 변환 (query_encode 벡터 크기 (1, 512) x hf_embeddings 벡터 크기 (1024,) 불일치해서 변환하였음)
    if len(numbers) == 512:
        return numbers
    else:
        # 512차원이 아닌 경우 영벡터로 반환
        return [0.0] * 512

# hf_embeddings 컬럼에 함수 적용
food_data['hf_embeddings'] = food_data['hf_embeddings'].apply(parse_embeddings)

# 쿼리와 가장 유사한 상위 k개의 항목을 찾는 함수 정의
def get_query_sim_top_k(query, model, df, top_k):
    # 쿼리를 벡터로 인코딩
    query_encode = model.encode(query).astype(np.float32)
    cos_scores = []
    # 데이터프레임의 각 항목과 쿼리 벡터 간의 코사인 유사도를 계산
    for a in df['hf_embeddings']:
        #a = a[:-1]
        #a = a[1:]
        #a = [float(ele) for ele in a.split()]
        cos_score = util.pytorch_cos_sim(query_encode, torch.tensor(a, dtype=torch.float32))
        cos_scores.append(cos_score)
    # 상위 k개의 유사도를 가진 결과 반환
    top_results = torch.topk(torch.tensor(cos_scores), k=top_k)
    return top_results

# ChatGPT 모델을 사용하여 메세지를 생성하는 함수 정의
def get_chatgpt_msg(msg):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msg
    )
    return completion['choices'][0]['message']['content']

#사용자 메세지 프롬프트 초기화
msg_prompt = {
    'recom' : {
                'system' : "You are a helpful assistant who recommends food items based on user question.",
                'user' : "Write 1 sentence of a simple greeting that starts with 'Of course!' to recommend food items to users.",
              },
    'desc' : {
                'system' : "You are a helpful assistant who kindly answers.",
                'user' : "Please write a simple greeting starting with 'Of course' to explain the item to the user.",
              },
    'intent' : {
                'system' : "You are a helpful assistant who understands the intent of the user's question.",
                'user' : "Which category does the sentence below belong to: 'description', 'recommended', 'search'? Show only categories. \n context:"
                }
}

# 사용자 메세지 기록 초기화
if 'user_msg_history' not in st.session_state:
    st.session_state.user_msg_history = []

# 사용자의 의도에 따라 적절한 프롬프트를 설정하는 함수 정의
def set_prompt(intent, query, msg_prompt_init, model):
    m = []
    if ('recom' in intent) or ('search' in intent):
        msg = msg_prompt_init['recom']
    elif 'desc' in intent:
        msg = msg_prompt_init['desc']
    else:
        msg = msg_prompt_init['intent']
        msg['user'] += f' {query} \n A:'
    for k, v in msg.items():
        m.append({'role': k, 'content': v})
    return m

# 알레르기 물질이 포함되지 않은 식품을 필터링하는 함수 정의
def filter_foods_by_allergens(allergens, food_data):
    def does_not_contain_allergens(allergens_text):
        if pd.isna(allergens_text):
            return True
        return all(allergen.lower() not in allergens_text.lower() for allergen in allergens)
    # 알레르기 물질이 포함되지 않은 행만 필터링
    filtered_food_data = food_data[food_data['allergens'].apply(does_not_contain_allergens)]
    return filtered_food_data


# 사용자의 쿼리에 대한 상호작용을 처리하는 함수 정의
def user_interact(query, model, msg_prompt_init, allergens=None):
    # 1. 사용자의 의도를 파악하기 위한 프롬프트 설정
    user_intent_prompt = set_prompt('intent', query, msg_prompt_init, None)
    # ChatGPT를 사용하여 사용자의 의도를 파악
    user_intent = get_chatgpt_msg(user_intent_prompt).lower().strip()
    
    # 2. 사용자의 의도에 따라 적절한 데이터 프롬프트 설정
    intent_data_prompt = set_prompt(user_intent, query, msg_prompt_init, model)
    # ChatGPT를 사용하여 응답 메세지 생성
    intent_data_msg = get_chatgpt_msg(intent_data_prompt).replace("\n", "").strip()

    # 3-1. 사용자의 의도가 추천 또는 검색인 경우
    if ('recom' in user_intent) or ('search' in user_intent):
        recom_msg = f"{intent_data_msg}\n\n" #포맷 변경! 엔터추가
        # 기존에 메세지가 있으면 쿼리로 대체
        if (len(st.session_state.user_msg_history) > 0) and (st.session_state.user_msg_history[-1]['role'] == 'assistant'):
            last_content = st.session_state.user_msg_history[-1]['content']
            if isinstance(last_content, dict) and 'feature' in last_content:
                query = last_content['feature']
        
        # 알레르기 유발 물질이 제공된 경우 필터링된 식품 데이터를 사용
        if allergens:
            filtered_food_data = filter_foods_by_allergens(allergens, food_data)
        else:
            filtered_food_data = food_data
        # 쿼리와 유사한 상위 결과를 가져옴
        top_result = get_query_sim_top_k(query, model, filtered_food_data, top_k=3)
        top_index = top_result[1].numpy() if 'recom' in user_intent else top_result[1].numpy()[1:]
        # 상위 결과를 데이터프레임에서 추출 - 식품 제품명, 브랜드, 나라, 재료, 알레르기 정보를 가져와서 출력
        r_set_d = filtered_food_data.iloc[top_index, :][['product_name', 'brands', 'countries', 'ingredients_text', 'allergens']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))
        
        # 결과 메세지 구성 # 포맷 변경! Top1,2,3 # ** bold 체
        for i, r in enumerate(r_set_d, start=1):
            recom_msg += f"**Top{i}**  \n"
            for k, v in r.items():
                recom_msg += f"{k}: {v}  \n"  # 두 개의 공백을 추가하여 줄바꿈을 명확히 함
                recom_msg += "\n"
        st.markdown(recom_msg)  # st.write 대신 st.markdown을 사용하여 Markdown 형식을 지원

        
    # 3-2. 사용자의 의도가 설명인 경우
    elif 'desc' in user_intent:
        # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
        top_result = get_query_sim_top_k(st.session_state.user_msg_history[-1]['content'], model, food_data, top_k=1)
        # feature가 상세 설명이라고 가정하고 해당 컬럼의 값을 가져와 출력
        r_set_d = food_data.iloc[top_result[1].numpy(), :][['feature']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))[0]
        st.write(f"{intent_data_msg} {r_set_d}")

# Streamlit 앱 레이아웃 설정
st.title("GPT-Based Food Recommend System")
query = st.text_input("Enter your query:")
allergens_input = st.text_input("Enter allergens to avoid (comma separated):")
if allergens_input:
    allergens = [allergen.strip() for allergen in allergens_input.split(",")]
else:
    allergens = []

# 'Submit' 버튼 클릭 시, 사용자의 쿼리를 처리하여 결과를 출력
if st.button("Submit"):
    user_interact(query, model, copy.deepcopy(msg_prompt), allergens)
    st.session_state.user_msg_history.append({'role': 'user', 'content': query})

# 'Refresh' 버튼 클릭 시, 사용자 메시지 기록 초기화
if st.button("Refresh"):
    st.session_state.user_msg_history = []
    st.write("History has been reset.")