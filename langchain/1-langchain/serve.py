from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import os
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile")

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "Translate the following from **{src}** to **{dest}**"),
    ("user", "{query}")]
)

chain = prompt_template | model | StrOutputParser()

# chain.invoke({"query": "How are you doing today??", "src": "English", "dest": "French"})

app = FastAPI(title="LangChain - LangServe", version="1.0", description="LangChain - LangServe")

add_routes(app, chain, path="/chain")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)