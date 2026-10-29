import os
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# 1. Load Environment Variables
load_dotenv()

# 2. Load Documents
pdf_files = [
    "sebi_financial_education.pdf",
    "gst_ready_reckoner.pdf",
]
all_documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    all_documents.extend(loader.load())

# 3. Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_documents)

# 4. Setup Chroma Vector Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="finance_docs",
    embedding_function=embeddings,
    persist_directory="./tmp/chroma_langchain_db",
)
vector_store.add_documents(documents=chunks)

# 5. Define PDF Search Tool
@tool 
def search_finance_docs(question: str) -> str:
    """Search official Indian government financial documents for rules, regulations, tax rates, investment guidance, and financial education content."""
    results = vector_store.similarity_search(question, k=3)
    
    if not results:
        return "No relevant information found in the knowledge base."
        
    context = ""
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) 
        context += f"Source: {source} (Page {page + 1})\n"
        context += f"Content: {doc.page_content}\n\n"
        
    return context

# 6. Define Live Market API Tool
@tool
def get_market_price(asset: str) -> str:
    """Get current market price for a financial asset in Indian Rupees. Pass one of: gold, silver."""
    symbols = {"gold": "XAU", "silver": "XAG"}
    symbol = symbols.get(asset.lower())
    
    if not symbol:
        return "Supported assets: gold, silver"

    url = f"https://api.gold-api.com/price/{symbol}/INR"       
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        price = data.get("price")
        
        if price:
            price_per_gram = price / 31.1035
            return f"{asset.title()}: ₹{price_per_gram:.2f} per gram (₹{price:.2f} per troy ounce). Source: Gold-API"
            
    return f"Could not fetch price for {asset}"

# 7. Initialize AI Model and Agent
api_key = os.environ["GEMINI_API_KEY"]

model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    api_key=api_key,
)

system_prompt = """You are a Personal Finance AI Advisor for Indian citizens.

You have access to these tools:
- search_finance_docs: Search official Indian government documents for GST rates, investment guidance, tax saving options, and financial education
- get_market_price: Get live market prices for gold and silver in Indian Rupees

Help the user by looking up the relevant data using your tools and giving clear, specific answers with actual numbers and rates. Always mention the source of your information. All monetary values should be in Indian Rupees (₹) unless specified otherwise.
"""

agent = create_agent(
    model=model,
    tools=[search_finance_docs, get_market_price],
    system_prompt=system_prompt,
)    

# 8. Terminal Testing
print("Testing Tool 1: Live API...")
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the current gold price?"}]}
)
print(response1["messages"][-1].text)
print("-" * 50)

print("Testing Dual Tools: API + PDF Search...")
response2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I want to buy a gold chain. What's the current gold price and how much GST will I pay?",
            }
        ]
    }
)
print(response2["messages"][-1].text)
print("-" * 50)

# 9. Gradio UI
def finance_advisor(question):
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].text

demo = gr.Interface(
    fn=finance_advisor,
    inputs=gr.Textbox(lines=2, placeholder="Ask a finance question...", label="Question"),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="Personal Finance AI Advisor",
    description="Ask about gold prices, silver prices, GST rates, tax saving options, and more.",
)

demo.launch(server_name="0.0.0.0", root_path="/gradio")