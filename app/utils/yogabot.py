import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from pathlib import Path

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
vectorDB = Path(__file__).parent / "data" / "yoga_vectorstore"

def initialize_retrievalchain():

    llm = Ollama(model="llama2",base_url=OLLAMA_HOST)

    embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST)

    yoga_vector = FAISS.load_local(vectorDB, embeddings, allow_dangerous_deserialization=True)

    prompt = ChatPromptTemplate.from_template("""You are a yoga guru and answer the following question based on your knowledge and the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = yoga_vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
