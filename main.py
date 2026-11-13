import json
from bs4 import BeautifulSoup
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

def load_sec_filing(ticker="SBUX"):
    json_url = "https://raw.githubusercontent.com/yhh1994/10q/main/data.json"
    print(f"Fetching filing registry from online repository...")
    response = requests.get(json_url)
    response.raise_for_status()
    filings = response.json()
    
    target = next((item for item in filings if item["ticker"] == ticker), None)
    if not target:
        raise ValueError(f"Ticker {ticker} not found in remote data.json")
        
    print(f"Fetching SEC filing for {target['title']} from {target['link']}...")
    
    headers = {"User-Agent": "FinancialRAGAgent davidye2027@gmail.com"}
    sec_response = requests.get(target["link"], headers=headers)
    sec_response.raise_for_status()
    
    soup = BeautifulSoup(sec_response.text, "html.parser")
    
    # 1. Remove non-content tags that pollute the text context
    for element in soup(["script", "style", "head", "meta", "link"]):
        element.decompose()
        
    # 2. Extract text with preserved line breaks to help maintain table row structures
    clean_text = soup.get_text(separator="\n", strip=True)
    return clean_text, target

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
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print(f"Total chunks: {len(docs)}")
    print(f"Longest chunk: {max(len(d.page_content) for d in docs)} characters")
    '''
    print("Testing embeddings on actual SEC chunks...")
    
    test_embeddings = embeddings.embed_documents(
        [doc.page_content for doc in docs[:10]]
    )
    
    print(f"Successfully embedded {len(test_embeddings)} chunks")
    '''
    print("Generating local embeddings with nomic-embed-text...")
    # Using your newly pulled embedding model
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Store locally in ChromaDB
    print("Creating ChromaDB vector store...")

    batch_size = 16
    
    vectorstore = Chroma(
        collection_name="sec_filings",
        embedding_function=embeddings,
        persist_directory="./chroma_db_test"
    )
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        '''
        print(
            f"Embedding and storing chunks "
            f"{i + 1}-{min(i + batch_size, len(docs))} "
            f"of {len(docs)}..."
        )
        '''
        vectorstore.add_documents(
            documents=batch,
            ids=[f"SBUX_10K_{j}" for j in range(i, i + len(batch))]
        )
    
    print("All SEC chunks successfully stored in ChromaDB!")
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
