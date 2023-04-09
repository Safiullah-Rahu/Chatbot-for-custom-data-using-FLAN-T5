"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

# From here down is all the StreamLit UI.
st.set_page_config(page_title="FLAN-T5 XXL Model", page_icon=":robot:")
st.header("FLAN-T5 XXL Model")

from langchain.chains import ConversationChain
from langchain.llms import HuggingFaceHub

import os
#os.environ["HUGGINGFACEHUB_API_TOKEN"]
flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature":0.1,
                 "max_new_tokens":256,
                 "min_tokens":200},
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    qa = ConversationChain(llm=flan_t5)
    return qa

qa = load_chain()



if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = qa.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
