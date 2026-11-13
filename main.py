import json
import os
from bs4 import BeautifulSoup
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def load_sec_filing(ticker="SBUX"):
    # Load your data.json file
    with open("data/data.json", "r") as f:
        filings = json.load(f)
        
    # Find the matching entry for the ticker
    target = next((item for item in filings if item["ticker"] == ticker), None)
    if not target:
        raise ValueError(f"Ticker {ticker} not found in data.json")
        
    print(f"Fetching SEC filing for {target['title']} from {target['link']}...")
    
    # SEC requires a User-Agent header or it will block the request
    headers = {"User-Agent": "MyFinancialAgent research@example.com"}
    response = requests.get(target["link"], headers=headers)
    response.raise_for_status()
    
    # Parse HTML and extract plain text
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(), target

def main():
    # 1. Load SEC Filing data from your local JSON mapping
    raw_text, filing_meta = load_sec_filing("SBUX")
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.create_documents(
        texts=[raw_text], 
        metadatas=[{"ticker": filing_meta["ticker"], "form": filing_meta["form"]}]
    )
    
    print("Creating local vector embeddings (using Ollama)...")
    # Uses Ollama for local embeddings (e.g., nomic-embed-text or llama3)
    embeddings = OllamaEmbeddings(model="llama3")
    
    # Store chunks locally in ChromaDB
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    print("Connecting to local Ollama LLM...")
    # Initialize the local model via LangChain
    llm = ChatOllama(model="llama3", temperature=0)
    
    # Setup Retrieval-QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Ask a financial question
    query = "What were the total net revenues for the fiscal year?"
    print(f"\nQuestion: {query}")
    
    response = qa_chain.invoke({"query": query})
    print(f"\nAnswer:\n{response['result']}")

if __name__ == "__main__":
    main()
