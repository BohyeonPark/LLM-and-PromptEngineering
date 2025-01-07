#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!pip install langchain-experimental
#!pip install tabulate
#!pip install pandas
#!pip install openai
#!pip install streamlit


# In[4]:


import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = "sk-"  # OpenAI API 키 입력

# Streamlit 웹페이지 설정
st.set_page_config(page_title="CSV 파일 분석 애플리케이션", layout="wide")

# 웹페이지 타이틀
st.title("CSV 파일 분석 애플리케이션")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
if uploaded_file is not None:
    # 파일을 데이터프레임으로 읽기
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터프레임:")
    st.dataframe(df.head())  # 처음 5개의 행을 미리 보기

    # LangChain 에이전트 생성
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
    )

    # 사용자가 입력한 질문을 받아서 처리
    user_input = st.text_input("분석할 질문을 입력하세요:", "")

    if st.button("질문 실행"):
        if user_input:
            with st.spinner("질문을 처리 중입니다..."):
                # 에이전트가 질문을 처리하도록 요청
                response = agent.run(user_input)
                st.write("응답:", response)
        else:
            st.warning("질문을 입력해주세요.")

else:
    st.info("CSV 파일을 업로드하여 분석을 시작하세요.")


# In[ ]:




