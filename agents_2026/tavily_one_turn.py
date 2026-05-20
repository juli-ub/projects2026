import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

# Load environment variables from .env file
load_dotenv()

# 1. Initialize Gemini model (Gemini 2.5 Flash works well for tool calling)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# 2. Setup the Tavily Search Tool using the updated 'langchain-tavily' package
# We explicitly name the tool "tavily_search" to match your system prompt
search_tool = TavilySearch(max_results=3, name="tavily_search")
tools = [search_tool]

# 3. Your default system prompt
default_prompt = (
    "You are a helpful assistant. Answer questions directly when you know the answer. "
    "If you need current or detailed information, call the `tavily_search` tool. "
    "Only call the tool when necessary. "
    "In your answer say if you used the tool and what info you got from it. Be concise. "
    "Otherwise, just answer the question based on your existing knowledge and say that you didn't need to use the tool."
)

# 4. Create the agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=default_prompt
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
def run_agent(question: str):
    print(f"\nUser: {question}")
    print("=" * 60)
    
    # We use agent.stream with stream_mode="updates" to capture each step of the loop
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
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
    # Prompt the user for a single input
    user_query = input("Ask a question: ")
    
    if user_query.strip():
        run_agent(user_query)
    else:
        print("No input provided. Exiting.")