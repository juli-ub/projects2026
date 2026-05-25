import os
import sqlite3
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent

# Global reference for our local FAISS index
vector_store = None


# 1. Setup the local SQLite Database with data records
def setup_database():
    """Creates a local database file and inserts sample inventory records."""
    conn = sqlite3.connect("company_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT UNIQUE,
            quantity INTEGER,
            price REAL
        )
    """)
    
    sample_items = [
        ("Quantum Widget", 12, 199.99),
        ("Nebula Laptop", 5, 1299.50),
        ("Cosmic Keyboard", 45, 89.00),
        ("HyperDrive Router", 2, 450.00)
    ]
    
    for name, qty, price in sample_items:
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO inventory (item_name, quantity, price) VALUES (?, ?, ?)",
                (name, qty, price)
            )
        except sqlite3.Error:
            pass
            
    conn.commit()
    conn.close()


# 2. Load SQLite records and index them into FAISS
def build_vector_store():
    """Reads inventory from SQLite and populates the local FAISS vector database."""
    global vector_store
    
    # Fetch rows from SQLite
    conn = sqlite3.connect("company_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT item_name, quantity, price FROM inventory")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert records into descriptive Documents for semantic matching
    documents = []
    for name, qty, price in rows:
        doc_content = f"Product: {name} | Stock: {qty} units | Price: ${price:.2f}"
        doc = Document(
            page_content=doc_content,
            metadata={"item_name": name, "quantity": qty, "price": price}
        )
        documents.append(doc)
        
    # Initialize the Google GenAI embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Load document vectors into our local in-memory FAISS database
    print("[System] Generating embeddings and building local FAISS store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("[System] Vector database successfully created.")


# 3. Define the Vector Store-based Tool
@tool
def query_inventory(item_name: str) -> str:
    """
    Queries the local inventory vector store to find product details, pricing, and stock levels.
    Use this tool only when asked about stock levels, quantities, prices, or product availability.
    """
    global vector_store
    if vector_store is None:
        return "Error: Inventory vector store is not initialized."
    
    # Query FAISS for the single most similar document (k=1)
    results = vector_store.similarity_search_with_score(item_name, k=1)
    
    if not results:
        return f"No matches found for '{item_name}'."
        
    doc, score = results[0]
    
    # Return the text representation back to the agent.
    # The agent's LLM will verify if the returned item actually matches the user's query.
    return doc.page_content


# Helper function to extract text content safely from different message formats
def get_clean_content(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


# 4. Define the Agent Execution Flow
def run_agent():
    # Set up database file
    setup_database()
    
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # Build local vector store using records loaded from the SQLite database
    build_vector_store()

    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Agent system prompt
    system_prompt = (
        "You are an assistant.\n"
        "If the user asks general questions, answer directly using your knowledge.\n"
        "If the user asks about specific product inventory, availability, stock, or pricing, "
        "you must use the 'query_inventory' tool. Do not guess the details. "
        "The tool uses a semantic vector search, so it will always return the most similar item. "
        "You must evaluate the tool's output; if the retrieved product does not match the item the user requested, "
        "inform them that we do not carry or have that item in stock."
    )

    # Create the agent
    agent = create_agent(
        model=llm,
        tools=[query_inventory],
        system_prompt=system_prompt
    )

    # --- Scenario 1: General Knowledge ---
    print("\n[Scenario 1: General Question]")
    q1 = "What causes solar eclipses?"
    print(f"User: {q1}")
    result1 = agent.invoke({"messages": [{"role": "user", "content": q1}]})
    print(f"Agent Response: {get_clean_content(result1['messages'][-1])}")

    # --- Scenario 2: Semantic Matching (FAISS should resolve "Laptops" to "Laptop") ---
    print("\n[Scenario 2: Database Query with Plural/Semantic terms]")
    q2 = "How many Nebula Laptops do we currently have, and how much are they?"
    print(f"User: {q2}")
    result2 = agent.invoke({"messages": [{"role": "user", "content": q2}]})
    print(f"Agent Response: {get_clean_content(result2['messages'][-1])}")

    # --- Scenario 3: Missing Item (Evaluates how the agent filters out false positives) ---
    print("\n[Scenario 3: Product Not in Database]")
    q3 = "Can you look up if we have solar chargers in stock?"
    print(f"User: {q3}")
    result3 = agent.invoke({"messages": [{"role": "user", "content": q3}]})
    print(f"Agent Response: {get_clean_content(result3['messages'][-1])}")


if __name__ == "__main__":
    run_agent()