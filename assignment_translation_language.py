#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install langchain
#!pip install streamlit
#!pip install openai
#!pip install langchain-community


# In[2]:


import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os
os.environ["OPENAI_API_KEY"] = "sk-" #openapi


# In[3]:


#웹페이지에 보여질 내용
langs = ["Korean", "Japanese", "chinese", "English"] #번역할 언어를 나열
left_co, cent_co,last_co = st.columns(3)
#웹페이지 왼쪽에 언어를 선택할 수 있는 라디오 버튼
with st.sidebar:
     language = st.radio('번역을 원하는 언어를 선택해주세요.:', langs)
st.markdown('### 언어 번역 서비스예요~')
prompt = st.text_input('번역을 원하는 텍스트를 입력하세요') #사용자의 텍스트입력
#프롬프트를 번역으로 지시
trans_template = PromptTemplate(
input_variables=['trans'],
template='Your task is to translate this text to ' + language + 'TEXT: {trans}'
)
#momory는 텍스트 저장 용도
memory = ConversationBufferMemory(input_key='trans', memory_key='chat_history')
llm = ChatOpenAI(temperature=0.0,model_name='gpt-4')
trans_chain = LLMChain(llm=llm, prompt=trans_template, verbose=True, output_key='translate', memory=memory)
#프롬프트(trans_template)가 있으면 이를 처리하고 화면에 응답을 작성
if st.button("번역"):
   if prompt:
      response = trans_chain({'trans': prompt})
      st.info(response['translate'])


# In[ ]:




