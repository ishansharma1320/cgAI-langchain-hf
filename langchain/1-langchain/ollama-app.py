import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_ollama import ChatOllama
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

if __name__ == '__main__':
    prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a helpful assistant. Please respond to the question asked to the best of your ability"),
     ("user","Question: {question}")
    ])

    st.title("Langchain Demo with Ollama")
    input_text = st.text_input("What question do you have in mind?")
    llm = ChatOllama(model="artifish/llama3.2-uncensored:latest")

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": input_text})
    st.write(f"{response}")
# ollama_embed = OllamaEmbeddings(model="nomic-embed-text:latest")
# db_ollama = Chroma.from_documents(split_documents, ollama_embed, collection_name='nomic_embed_ollama_embed')