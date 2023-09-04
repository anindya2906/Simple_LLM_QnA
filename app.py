"""Main Application module
"""

import streamlit as st
from langchain.llms import OpenAI, HuggingFaceHub


def get_user_question():
    """Get user input
    """
    user_input = st.text_input(
        "Question: ", key="user_input", label_visibility="hidden")
    return user_input


def get_answer(question):
    """Get answer
    """
    return llm(question)


st.set_page_config(
    page_title="Simple QnA with LLM (Langchain)", page_icon=":robot:")
st.header("Simple QnA with LLM (Langchain)")

llm_provider = st.selectbox("LLM: ", ("OpenAI", "HuggingFace HUB"))

if llm_provider == "OpenAI":
    llm = OpenAI(model="text-davinci-003", temperature=0.0)
else:
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                         model_kwargs={"temperature": 0.5})

st.divider()

st.subheader("Question:")
user_question = get_user_question()

generate = st.button("Generate")

if generate and user_question:
    answer = get_answer(user_question)
    st.subheader("Answer:")
    st.write(answer)
