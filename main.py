
import hashlib
import re
import requests

from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA


JSON_URL = "https://raw.githubusercontent.com/yhh1994/10q/main/data.json"
DB_DIRECTORY = "./chroma_db_incremental"
COLLECTION_NAME = "sec_filings"
BATCH_SIZE = 16


def get_accession_number(filing_url):
    """Extract and format the SEC accession number from the filing URL."""
    match = re.search(
        r"/Archives/edgar/data/\d+/(\d{18})/",
        filing_url
    )

    if not match:
        raise ValueError(
            f"Could not extract SEC accession number from: {filing_url}"
        )

    accession = match.group(1)

    # SEC accession number format: 0000829224-26-000162
    return f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"


def load_sec_filing(ticker="SBUX", form="10-K"):
    """Find and download the latest filing for a ticker and form."""

    print("Fetching filing registry from online repository...")

    response = requests.get(JSON_URL, timeout=30)
    response.raise_for_status()
    filings = response.json()

    # Find all filings matching the requested ticker and form.
    matches = [
        item for item in filings
        if item["ticker"].upper() == ticker.upper()
        and item["form"] == form
    ]

    if not matches:
        raise ValueError(
            f"No {form} filing found for ticker {ticker}"
        )

    # Select the most recent filing based on the registry date.
    target = max(matches, key=lambda item: item["date"])

    accession_number = get_accession_number(target["link"])

    print(f"Fetching filing: {target['title']}")
    print(f"Filing date: {target['date']}")
    print(f"Accession number: {accession_number}")

    headers = {
        "User-Agent": "FinancialRAGAgent davidye2027@gmail.com"
    }

    sec_response = requests.get(
        target["link"],
        headers=headers,
        timeout=60
    )
    sec_response.raise_for_status()

    soup = BeautifulSoup(sec_response.text, "html.parser")

    # Remove non-content elements.
    for element in soup(["script", "style", "head", "meta", "link"]):
        element.decompose()

    clean_text = soup.get_text(separator="\n", strip=True)

    return clean_text, target, accession_number


def create_documents(raw_text, filing_meta, accession_number):
    """Split filing text and attach metadata to every chunk."""

    print("Splitting SEC document into manageable chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Hash the extracted filing text to detect content changes.
    document_hash = hashlib.sha256(
        raw_text.encode("utf-8")
    ).hexdigest()

    base_metadata = {
        "ticker": filing_meta["ticker"],
        "form": filing_meta["form"],
        "filing_date": filing_meta["date"],
        "accession_number": accession_number,
        "filing_title": filing_meta["title"],
        "filing_url": filing_meta["link"],
        "document_hash": document_hash
    }

    docs = text_splitter.create_documents(
        texts=[raw_text],
        metadatas=[base_metadata]
    )

    # Add chunk-level metadata.
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["total_chunks"] = len(docs)

    print(f"Total chunks: {len(docs)}")
    print(
        f"Longest chunk: "
        f"{max(len(doc.page_content) for doc in docs)} characters"
    )

    return docs, document_hash


def update_vectorstore(vectorstore, docs, accession_number, document_hash):
    """
    Check whether a filing already exists.

    - Unchanged filing: reuse existing embeddings.
    - Changed filing: delete old chunks and re-ingest.
    - New filing: embed and store in batches.
    """

    print("Checking ChromaDB for existing filing...")

    existing = vectorstore.get(
        where={"accession_number": accession_number},
        include=["metadatas"]
    )

    existing_ids = existing["ids"]
    existing_metadata = existing.get("metadatas") or []

    # Check whether all expected chunks are already stored.
    unchanged = (
        len(existing_ids) == len(docs)
        and all(
            metadata.get("document_hash") == document_hash
            for metadata in existing_metadata
        )
    )

    if unchanged:
        print(
            f"Filing {accession_number} is already stored "
            f"with {len(existing_ids)} chunks."
        )
        print("Filing unchanged. Reusing existing embeddings.")

        return

    # If an older or partially ingested version exists, replace it.
    if existing_ids:
        print(
            f"Existing filing found with {len(existing_ids)} chunks, "
            "but its content differs or ingestion is incomplete."
        )
        print("Removing old chunks before re-ingestion...")

        vectorstore.delete(ids=existing_ids)

    else:
        print("New filing detected. Starting ingestion...")

    # Embed and store in small batches to avoid Ollama EOF errors.
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]

        end_index = min(i + len(batch), len(docs))

        print(
            f"Embedding and storing chunks "
            f"{i + 1}-{end_index} of {len(docs)}..."
        )

        # Stable IDs make individual chunks identifiable.
        chunk_ids = [
            f"{accession_number.replace('-', '')}_{j}"
            for j in range(i, end_index)
        ]

        vectorstore.add_documents(
            documents=batch,
            ids=chunk_ids
        )

    print(
        f"Successfully stored {len(docs)} chunks "
        f"for accession {accession_number}."
    )


def main():

    ticker = "SBUX"
    form = "10-K"

    # 1. Download the latest filing.
    raw_text, filing_meta, accession_number = load_sec_filing(
        ticker=ticker,
        form=form
    )

    # 2. Split the filing and generate metadata.
    docs, document_hash = create_documents(
        raw_text,
        filing_meta,
        accession_number
    )

    # 3. Initialize local embeddings.
    print("Initializing local embeddings...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # 4. Connect to persistent ChromaDB.
    print("Connecting to ChromaDB...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIRECTORY
    )

    # 5. Check for existing filing and ingest only if needed.
    update_vectorstore(
        vectorstore,
        docs,
        accession_number,
        document_hash
    )

    # 6. Retrieve relevant filing chunks.
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"ticker": ticker}
        }
    )

    # 7. Initialize local LLM.
    print("Initializing local LLM with llama3...")

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 8. Ask a financial question.
    query = "What were the total net revenues for the fiscal period?"

    print(f"\nQuerying RAG agent: '{query}'")

    response = qa_chain.invoke({
        "query": query
    })

    print(f"\nAnswer:\n{response['result']}")

    # Show the source chunks used for the answer.
    print("\nSource documents:")

    for doc in response["source_documents"]:
        print(
            f"- {doc.metadata['ticker']} | "
            f"{doc.metadata['form']} | "
            f"{doc.metadata['filing_date']} | "
            f"Chunk {doc.metadata['chunk_index']}"
        )


if __name__ == "__main__":
    main()
