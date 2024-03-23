from prompt_templates import memory_prompt_template, pdf_chat_prompt
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from operator import itemgetter
from utils import *
import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_llm():
    load_dotenv()
    llm = ChatOpenAI()
    return llm

def create_embeddings():
    embeddings=OpenAIEmbeddings()
    return embeddings

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_history=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm, chat_prompt, memory)

def load_normal_chain(chat_history):
    return chatChain(chat_history)


class chatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = LLMChain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input = user_input,history=self.memory.chat_memory.messages, stop=["Human:"])








