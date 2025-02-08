#%%
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from fastapi import FastAPI
from langserve import add_routes
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#%%
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"] = "QA Chatbot Conversational RAG"
os.environ["HF_TOKEN"]=os.getenv("HUGGINGFACE_API_KEY")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

#%%

st.title("Talk to your PDF - Upload PDF and chat history - Conversational RAG")
st.write("Upload PDF and chat with your content")
groq_llm = ChatGroq(model="gemma2-9b-it")
embed_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

session_id = st.text_input("Session ID: ", value="default")
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    os.makedirs("pdfs", exist_ok=True)
    for uploaded_file in uploaded_files:
        root_path = os.path.join(os.getcwd(), "pdfs")
        file_path = os.path.join(root_path,uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
    
    print(f"Number of documents loaded: {len(documents)}")
    for doc in documents:
        print(f"Document content: {doc.page_content[:500]}")  # Print the first 500 characters

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    db_faiss = FAISS.from_documents(split_docs,embed_hf)
    retriever = db_faiss.as_retriever()
    system_contextualises_user_query = """Given the user query and chat history, 
    The chat history may have the reference context behind the user query
    Formulate a stand alone query which can be understood

    ** DO NOT ANSWER THE USER QUERY **

    Only reformulation is needed.
    The use of this query is to make a semantic search to a database
    If you don't find any relevant information, return the query as is."""
    prompt_that_contextualises_user_query = ChatPromptTemplate.from_messages(
    [("system", system_contextualises_user_query),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")
     ]
)

    history_aware_retriever = create_history_aware_retriever(llm=groq_llm, retriever=retriever,prompt=prompt_that_contextualises_user_query)
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
    with_message_history = RunnableWithMessageHistory(crag_chain, get_session_history, 
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer")

    user_input = st.text_input("Enter your question: ")
    if user_input:
        session_history = get_session_history(session_id)
        response = with_message_history.invoke({"input": user_input}, config={"configurable":{"session_id": session_id}})

        st.write(st.session_state.store)
        st.success(f"Assistant: {response['answer']}")
        st.write("Chat History: ", session_history.messages)
    else:
        st.error("No User Input!!")

else:
    st.error("No uploaded Files!!")
