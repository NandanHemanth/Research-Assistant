import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import *

# Takes PDF file, enumerates each page, extracs raw text and returns the concatenated text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Takes raw text as input, defines text_splitter with custom parameters and then returns split chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Takes the text_chunks as parameter, using OpenAI creates embeddings of the chunks, saves it in a vectorestore and returns vectorstore
def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore=FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

# Function to define a conversation chain and define OpenAI chat model and return this chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Handles User_Input with and displays with UI
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to run streamlit app - Functionality wrapped with Streamlit UI
def main():
    load_dotenv()
    st.set_page_config(page_title="Research Bot", page_icon="🤖")

    st.write(css, unsafe_allow_html=True)

    st.title("Research Assistant📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Type your message here:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vectorstore 
                vectorstore = get_vectorstore(text_chunks)
                # st.write(vectorstore)

                st.write("Documents processed successfully!")

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()