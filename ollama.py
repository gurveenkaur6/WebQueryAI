from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

# streamlit framework

st.title('Langchain Demo With Ollama(model-llama2) API')
input_text=st.text_input("Search the topic u want")

# using ollama as the llm. 
# To download the llama2 model, we need to write 'ollama run gemma' or llama2 or whatevre model you want on the terminal and 
# it will pull the desired model to your local.
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

# give the input as a question/prompt to the llm model
if input_text:
    st.write(chain.invoke({'question':input_text}))
