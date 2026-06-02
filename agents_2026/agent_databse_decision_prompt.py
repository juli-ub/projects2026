import os
import sqlite3
import datetime
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# 1. Setup the local SQLite Database with historical data
def setup_database():
    """Creates a local database file and inserts the initial record representing the default location."""
    conn = sqlite3.connect("company_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_statement TEXT UNIQUE,
            timestamp TEXT
        )
    """)
    
    # Clean up any past runs
    cursor.execute("DELETE FROM statements")
    
    # Default record: James lives in Paris (simulating a record from 2 years ago: 2024)
    # The current year is 2026, so June 2024 represents 2 years ago.
    past_timestamp = "2024-06-02 12:00:00"
    cursor.execute(
        "INSERT INTO statements (user_statement, timestamp) VALUES (?, ?)",
        ("James lives in Paris", past_timestamp)
    )
    
    conn.commit()
    conn.close()


# 2. Define the database tools for the agent
@tool
def save_statement(statement: str) -> str:
    """
    Saves a new statement about where someone currently lives to the database.
    This automatically applies the current system timestamp.
    Use this when the user directly states a new present fact (e.g., 'James lives in Berlin').
    """
    conn = sqlite3.connect("company_data.db")
    cursor = conn.cursor()
    
    # Current timestamp represents "now" (2026)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO statements (user_statement, timestamp) VALUES (?, ?)",
            (statement, current_time)
        )
        conn.commit()
        conn.close()
        return f"Successfully saved: '{statement}' with current timestamp {current_time}."
    except sqlite3.Error as e:
        conn.close()
        return f"Database error occurred: {str(e)}"


@tool
def get_statements() -> str:
    """
    Retrieves all stored location statements from the database along with their timestamps.
    Use this when you need to check past history or compare timelines.
    """
    conn = sqlite3.connect("company_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_statement, timestamp FROM statements")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No statements found in the database."
    
    formatted_rows = []
    for statement, timestamp in rows:
        formatted_rows.append(f"Statement: '{statement}' | Timestamp: {timestamp}")
        
    return "\n".join(formatted_rows)


# Helper function to extract text content safely from the final messages
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


# 3. Define the Execution Flow
def run_agent():
    # Setup initial database
    setup_database()

    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # Initialize Gemini 2.5 Flash Lite
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # System Prompt instructing the agent on database interactions and chronological logic
    system_prompt = (
       "You are an assistant specialized in tracking historical and current facts about entities over time.\n"
    "The current system date is June 2, 2026.\n\n"
    "Guidelines:\n"
    "1. When the user asserts a new, present-tense fact about an entity's status, condition, or location, "
    "use the 'save_statement' tool to record it in the database with the current timestamp.\n"
    "2. When the user queries the current status of an entity or asks about its historical timeline, "
    "use the 'get_statements' tool to retrieve all stored historical records.\n"
    "3. Perform chronological reasoning to resolve any conflicting information:\n"
    "   - Read and evaluate the recorded timestamps for all facts retrieved from the database.\n"
    "   - For any relative timeframes mentioned in the query (e.g., 'X years ago'), calculate the implied "
    "historical date relative to the current system date.\n"
    "   - Compare all evaluated dates. The statement with the most recent (newest) timestamp represents "
    "the current truth.\n"
    "4. Always present your timeline analysis step-by-step to the user before stating the final conclusion."
    )

    # Instantiate the agent
    agent = create_agent(
        model=llm,
        tools=[save_statement, get_statements],
        system_prompt=system_prompt
    )

    # --- Scenario 1: User provides new location statement ---
    print("\n[Scenario 1: Saving New Statement]")
    q1 = "James lives in Berlin"
    print(f"User: {q1}")
    
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": q1}]
    })
    print(f"Agent Action/Response: {get_clean_content(result1['messages'][-1])}")

    # --- Scenario 2: Chronological evaluation ---
    print("\n[Scenario 2: Chronological Reasoning]")
    q2 = "5 years ago James lived in Barcelona, does he still live there?"
    print(f"User: {q2}")
    
    result2 = agent.invoke({
        # We append q2 to the context of the previous run to simulate conversational turn
        "messages": result1["messages"] + [{"role": "user", "content": q2}]
    })
    print(f"Agent Response:\n{get_clean_content(result2['messages'][-1])}")


if __name__ == "__main__":
    run_agent()