import os
import sqlite3
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

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
    
    # Mock records simulating our internal business database
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


# 2. Define the SQLite Database Tool with flexible token-based matching
@tool
def query_inventory(item_name: str) -> str:
    """
    Queries the internal company database to check current inventory stock and pricing.
    Use this tool only when asked about stock levels, quantities, prices, or product availability.
    """
    conn = sqlite3.connect("company_data.db")
    cursor = conn.cursor()
    
    # Fetch all items to perform flexible word-level matching in Python
    cursor.execute("SELECT item_name, quantity, price FROM inventory")
    rows = cursor.fetchall()
    conn.close()
    
    def normalize(word: str) -> str:
        w = word.lower().strip()
        # Basic plural-to-singular normalization
        if w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
            w = w[:-1]
        return w

    # Tokenize and normalize the incoming query (e.g., 'Nebula Laptops' -> {'nebula', 'laptop'})
    query_words = {normalize(w) for w in item_name.split()}
    
    matched_results = []
    for db_name, quantity, price in rows:
        db_words = {normalize(w) for w in db_name.split()}
        
        # Match if the normalized query words are contained within the database name words
        if query_words.issubset(db_words) or db_words.issubset(query_words) or any(qw in db_words for qw in query_words):
            matched_results.append(f"Product: {db_name} | Stock: {quantity} | Price: ${price:.2f}")
    
    if not matched_results:
        return f"No products found matching '{item_name}'."
    
    return "\n".join(matched_results)


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


# 3. Define the Agent Execution Flow
def run_agent():
    # Set up database file
    setup_database()

    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # Initialize Gemini model (using langchain-google-genai)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Instructions defining when the agent should answer directly vs query the DB tool
    system_prompt = (
        "You are an assistant.\n"
        "If the user asks general questions (e.g., science, history, programming), "
        "answer directly using your pre-trained knowledge.\n"
        "If the user asks about our specific product inventory, availability, stock, or pricing, "
        "you do not know this information by default. You must use the 'query_inventory' tool. "
        "Do not guess the quantities or prices."
    )

    # Use the modern create_agent construct from langchain.agents
    agent = create_agent(
        model=llm,
        tools=[query_inventory],
        system_prompt=system_prompt
    )

    # --- Scenario 1: General Knowledge (No SQLite Query Needed) ---
    print("\n[Scenario 1: General Question]")
    q1 = "What causes solar eclipses?"
    print(f"User: {q1}")
    
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": q1}]
    })
    print(f"Agent Response: {get_clean_content(result1['messages'][-1])}")

    # --- Scenario 2: Specific Database Query (Agent should query SQLite) ---
    print("\n[Scenario 2: Database Query]")
    q2 = "How many Nebula Laptops do we currently have, and how much are they?"
    print(f"User: {q2}")
    
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": q2}]
    })
    print(f"Agent Response: {get_clean_content(result2['messages'][-1])}")

    # --- Scenario 3: Database query for missing data ---
    print("\n[Scenario 3: Product Not in Database]")
    q3 = "Can you look up if we have solar chargers in stock?"
    print(f"User: {q3}")
    
    result3 = agent.invoke({
        "messages": [{"role": "user", "content": q3}]
    })
    print(f"Agent Response: {get_clean_content(result3['messages'][-1])}")


if __name__ == "__main__":
    run_agent()