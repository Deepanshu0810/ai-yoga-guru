import streamlit as st
import random
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.yogabot import initialize_retrievalchain

retrieval_chain = initialize_retrievalchain()

def response_generator(prompt):
    
    response = retrieval_chain.invoke({"input": prompt})
    response = response["answer"]

    for char in response:
        yield char
        time.sleep(0.01)


st.title("Yoga Guru")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role'],avatar=message['avatar']):
        st.markdown(message['text'])

if prompt := st.chat_input("Ask your guru ..."):
    with st.chat_message("You",avatar="👨"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "You", "avatar": "👨","text": prompt})

    with st.chat_message("Yoga Guru",avatar="🧘"):
        response = st.write_stream(response_generator(prompt))

    st.session_state.messages.append({"role": "Yoga Guru", "avatar": "🧘","text": response})