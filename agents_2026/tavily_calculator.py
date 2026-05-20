import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool  # Added to create the custom tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables from .env file
load_dotenv()

# 1. Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# 2. Define custom calculator tool
@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical calculations (add or subtract) on two numbers.

    Args:
        operation: The operation to perform. Must be either 'add' or 'subtract'.
        a: The first number.
        b: The second number.
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    else:
        raise ValueError("Invalid operation. Supported operations are 'add' and 'subtract'.")

# 3. Setup tools list
search_tool = TavilySearch(max_results=3, name="tavily_search")
tools = [search_tool, calculator]

# 4. Initialize Memory (Checkpointer)
memory = InMemorySaver()

# 5. Adjusted system prompt to include the new tool
default_prompt = (
    "You are a helpful assistant. Answer questions directly when you know the answer. "
    "If you need current or detailed information, call the `tavily_search` tool. "
    "If you need to perform addition or subtraction, call the `calculator` tool. "
    "Only call tools when necessary. "
    "In your answer say if you used a tool and what info you got from it. Be concise. "
    "Otherwise, just answer the question based on your existing knowledge and say that you didn't need to use any tools."
)

# 6. Create the agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=default_prompt,
    checkpointer=memory
)

# Helper function to extract text values from potential block-based list structures
def get_clean_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "".join(texts)
    return str(content)

# Helper function to run the agent, streaming intermediate steps
def run_agent(question: str, config: dict):
    print(f"\nUser: {question}")
    print("=" * 60)
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates"
    ):
        for node, data in chunk.items():
            print(f"\n[Step: {node}]")
            if "messages" in data:
                last_msg = data["messages"][-1]
                
                # Check and print tool calls if the model requested one
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        print(f"⚙️ Action: Calling '{tool_call['name']}' with args: {tool_call['args']}")
                
                # Print text content cleanly
                text_content = get_clean_text(last_msg.content)
                if text_content:
                    print(text_content)
                    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_session_2"}}
    
    print("--- Running Multi-Turn Conversation (Max 3 turns) ---")
    print("The agent now has access to search and calculation tools.")
    
    for turn in range(1, 4):
        user_query = input(f"\n[Turn {turn}/3] Ask a question: ")
        
        if not user_query.strip():
            print("No input provided. Ending session.")
            break
            
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Ending session.")
            break
            
        run_agent(user_query, config)