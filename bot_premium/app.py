import streamlit as st
from dotenv import load_dotenv
from langchain.memory import StreamlitChatMessageHistory
from llm_chains import *

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_filed():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input=""

def set_send_input():
    st.session_state.send_input = True
    clear_input_filed()

# Main Function
def main():
    load_dotenv()
    st.set_page_config("Research Bot", page_icon="🤖")
    st.title("Research Assistant📚")

    chat_container = st.container()

    if "send_input" not in st.session_state:
        st.session_state.send_input=False
        st.session_state.user_question=""

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("type your message here", key="user_input", on_change=set_send_input)

    send_button = st.button("Send", key="send_button")
    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":

            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                llm_response = llm_chain.run(st.session_state.user_question)
                st.chat_message("ai").write(llm_response)

if __name__ == "__main__":
    main()
                    