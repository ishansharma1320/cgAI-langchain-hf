import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from fastapi import FastAPI
from langserve import add_routes

#%%
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"] = "QA Chatbot Conversational RAG"
os.environ["HF_TOKEN"]=os.getenv("HUGGINGFACE_API_KEY")

#%%
loader = WebBaseLoader(
    web_paths=("https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer("div",attrs={"class": "theme-doc-markdown markdown"}))
)

docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)
embed_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db_faiss = FAISS.from_documents(split_documents,embed_hf)
faissRetriever = db_faiss.as_retriever()

#%%

system_contextualises_user_query = ("Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )
prompt_that_contextualises_user_query = ChatPromptTemplate.from_messages(
    [("system", system_contextualises_user_query),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")
     ]
)
groq_llm = ChatGroq()
history_aware_retriever = create_history_aware_retriever(llm=groq_llm, retriever=faissRetriever,prompt=prompt_that_contextualises_user_query)
#%%

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

qa_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")
     ]
)

stuff_docs_chain = create_stuff_documents_chain(groq_llm, qa_prompt)
crag_chain = create_retrieval_chain(history_aware_retriever, stuff_docs_chain)
#%%

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(crag_chain, get_session_history, 
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer")
#%%

app = FastAPI(title="LangChain - Talk to your document", version="1.0", description="LangChain - Talk to your document")

add_routes(app, with_message_history, path="/chain")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)

