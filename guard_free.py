from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = FastAPI(title="Maya - Compliance Guardian")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


llm = OllamaLLM(model="llama3.2:3b", temperature=0.15)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

loader = DirectoryLoader('compliance_docs', glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(k=4)


class Request(BaseModel):
    prompt: str


@app.post("/guard")
async def guard(request: Request):
    context_docs = retriever.invoke(request.prompt)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    template = ChatPromptTemplate.from_template("""
You are **Maya**, a friendly and professional Compliance Guardian...

**Behavior:**
- Casual chat → Reply normally
- Work-related (payments, code, customer data, etc.) → Give helpful answer + compliance guidance

Context: {context}
User Request: {prompt}
""")

    chain = template | llm
    result = chain.invoke({"context": context, "prompt": request.prompt})

    return {"compliance_guard": result.content if hasattr(result, 'content') else str(result)}


if __name__ == "__main__":
    print("🌟 Maya is running!")
    print("→ Go to: http://localhost:8000/static/index.html")