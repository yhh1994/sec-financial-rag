import json
from bs4 import BeautifulSoup
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def load_sec_filing(ticker="SBUX"):
    # Use the Raw GitHub URL format for data.json
    json_url = "https://raw.githubusercontent.com/yhh1994/10q/main/data.json"
    
    print(f"Fetching filing registry from online repository...")
    response = requests.get(json_url)
    response.raise_for_status()
    filings = response.json()
    
    # Find the matching entry for the ticker
    target = next((item for item in filings if item["ticker"] == ticker), None)
    if not target:
        raise ValueError(f"Ticker {ticker} not found in remote data.json")
        
    print(f"Fetching SEC filing for {target['title']} from {target['link']}...")
    
    # SEC requires a descriptive User-Agent header
    headers = {"User-Agent": "FinancialRAGAgent research@example.com"}
    sec_response = requests.get(target["link"], headers=headers)
    sec_response.raise_for_status()
    
    soup = BeautifulSoup(sec_response.text, "html.parser")
    return soup.get_text(), target

def main():
    # 1. Load filing based on your JSON mapping
    raw_text, filing_meta = load_sec_filing("SBUX")
    
    print("Splitting SEC document into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.create_documents(
        texts=[raw_text], 
        metadatas=[{"ticker": filing_meta["ticker"], "form": filing_meta["form"]}]
    )
    
    print("Generating local embeddings with nomic-embed-text...")
    # Using your newly pulled embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Store locally in ChromaDB
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    print("Initializing local LLM with llama3...")
    llm = ChatOllama(model="llama3", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    query = "What were the total net revenues for the fiscal period?"
    print(f"\nQuerying RAG agent: '{query}'")
    
    response = qa_chain.invoke({"query": query})
    print(f"\nAnswer:\n{response['result']}")

if __name__ == "__main__":
    main()
