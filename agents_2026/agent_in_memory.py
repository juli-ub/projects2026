import os
import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define a Custom SQLite Memory Manager
class SQLiteMemory:
    def __init__(self, db_path=":memory:"):
        """
        Initializes an in-memory SQLite database.
        Use a file path like 'chat_history.db' if you want persistent storage on disk.
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Creates a simple schema to store the turn history."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                response TEXT
            )
        """)
        self.conn.commit()

    def save_interaction(self, prompt: str, response: str):
        """Inserts the user's prompt and LLM's response into the database."""
        self.cursor.execute(
            "INSERT INTO conversation (prompt, response) VALUES (?, ?)",
            (prompt, response)
        )
        self.conn.commit()

    def get_all_context(self) -> str:
        """Retrieves and formats all past turns as a context string."""
        self.cursor.execute("SELECT prompt, response FROM conversation")
        rows = self.cursor.fetchall()
        
        context = ""
        for prompt, response in rows:
            context += f"User: {prompt}\nAI: {response}\n\n"
        return context.strip()

    def close(self):
        self.conn.close()


# 2. Main Agent Execution Flow
def run_agent():
    # Retrieve the Google API key from the environment variables
    # export GOOGLE_API_KEY="your-key" (Linux/macOS) or set GOOGLE_API_KEY="your-key" (Windows)
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # Initialize our custom SQLite memory in memory (:memory:)
    memory = SQLiteMemory()

    # Initialize the Gemini model via LangChain's Google Gen AI integration
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Define a prompt template that accepts the history retrieved from SQLite
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following conversation history to answer the user's current question:\n\n{history}"),
        ("human", "{question}")
    ])

    # Construct the simple LangChain pipeline
    chain = prompt_template | llm

    print("--- Starting Gemini Agent Simulation ---")

    # Turn 1: Provide information
    q1 = "Hi, my name is Alice and my favorite color is emerald green."
    print(f"User: {q1}")
    
    # Retrieve any history (will be empty for Turn 1)
    history = memory.get_all_context()
    
    # Run the model
    response1 = chain.invoke({"history": history, "question": q1})
    print(f"AI: {response1.content}\n")

    # Save prompt and answer into the SQLite DB
    memory.save_interaction(q1, response1.content)

    # Turn 2: Ask a follow-up question that relies on the context
    q2 = "Can you remind me what my name is and what color I like?"
    print(f"User: {q2}")

    # Retrieve the context from SQLite (now contains Turn 1)
    history = memory.get_all_context()
    
    # Run the model with the retrieved context
    response2 = chain.invoke({"history": history, "question": q2})
    print(f"AI: {response2.content}\n")

    # Clean up the database connection
    memory.close()

if __name__ == "__main__":
    run_agent()