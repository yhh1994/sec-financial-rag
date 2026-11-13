import json
from langchain_community.chat_models import ChatOllama

# Load your SEC filing links from the repository
with open("data/data.json", "r") as f:
    sec_filings = json.load(f)

# Example: Accessing Starbucks filing link from your JSON
sbux_filing = next(item for item in sec_filings if item["ticker"] == "SBUX")
print(f"Loading filing for {sbux_filing['ticker']} from: {sbux_filing['link']}")

# Initialize your open-weight model
llm = ChatOllama(model="llama3", temperature=0)
