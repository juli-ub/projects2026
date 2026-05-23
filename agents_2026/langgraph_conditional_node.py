import os
import re
from typing import TypedDict, List
from dotenv import load_dotenv

# Import LangChain & LangGraph components
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is present
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY in your .env file.")

# Initialize the Gemini model (gemini-2.5-flash is fast and cost-effective)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# =====================================================================
# 1. DEFINE STATE
# =====================================================================
class AgentState(TypedDict):
    query: str
    plan: str
    current_step: str
    messages: List[dict]
    loop_count: int
    calc_result: str
    search_result: str
    final_answer: str

# =====================================================================
# 2. DEFINE NODES
# =====================================================================

def start_node(state: AgentState):
    print("\n--- START NODE ---")
    return {
        "query": state["query"],
        "messages": [{"role": "user", "content": state["query"]}],
        "loop_count": 0
    }

def plan_node(state: AgentState):
    print("\n--- PLAN NODE ---")
    loop_count = state.get("loop_count", 0) + 1
    query = state["query"]
    messages = state["messages"]
    
    prompt = f"""You are an AI Planner. Your job is to construct a clear, single-sentence plan to answer the user query: "{query}"
    
    History of tools/information gathered so far:
    {messages}
    
    Choose your action carefully:
    - If a calculation is needed, plan to use the 'calculator'.
    - If you need recent/external world information, plan to use 'search'.
    - If you already have all the information required to formulate the final answer, plan to 'finish'."""
    
    response = llm.invoke(prompt)
    plan_text = response.content.strip()
    print(f"Plan: {plan_text}")
    
    return {
        "plan": plan_text,
        "loop_count": loop_count,
        "messages": messages + [{"role": "assistant", "content": f"Plan: {plan_text}"}]
    }

def router_node(state: AgentState):
    print("\n--- ROUTER NODE ---")
    plan_text = state.get("plan", "")
    loop_count = state.get("loop_count", 0)
    
    # Prevent infinite loops (limit of 3 cycles)
    if loop_count >= 3:
        print("⚠️ Loop limit reached! Forcing state machine to finish.")
        return {"current_step": "finish"}
        
    prompt = f"""Based on this plan: "{plan_text}"
    Choose exactly one next action:
    - 'calc' (if the plan requires evaluating a mathematical expression)
    - 'search' (if the plan requires looking up info on the internet)
    - 'finish' (if the plan is complete or you have enough info)
    
    Respond with ONLY one of these exact words: 'calc', 'search', or 'finish'."""
    
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    
    # Basic validation to clean up LLM output
    if "calc" in decision:
        decision = "calc"
    elif "search" in decision:
        decision = "search"
    else:
        decision = "finish"
        
    print(f"Decision: '{decision}' (Iteration: {loop_count}/3)")
    return {"current_step": decision}

def calculator_node(state: AgentState):
    print("\n--- CALCULATOR NODE ---")
    query = state["query"]
    
    prompt = f"""Extract only the mathematical expression from the user query or planning history. 
    Query: '{query}'
    History: {state['messages']}
    
    Do not include any text in your response. Only output numbers and basic operators (+, -, *, /). 
    Example: '142857 multiplied by 6' -> '142857 * 6'."""
    
    response = llm.invoke(prompt)
    expression = response.content.strip()
    
    # Safe regex evaluation to protect local system
    try:
        if re.match(r"^[\d\s+\-*/().]+$", expression):
            result = str(eval(expression))
        else:
            result = f"Error: Untrusted characters in math expression: {expression}"
    except Exception as e:
        result = f"Error evaluating expression '{expression}': {e}"
        
    print(f"Result: {expression} = {result}")
    return {
        "calc_result": result,
        "messages": state["messages"] + [{"role": "system", "content": f"Calculator output: {result}"}]
    }

def search_node(state: AgentState):
    print("\n--- SEARCH NODE ---")
    query = state["query"]
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key or api_key == "your_tavily_api_key_here":
        print("Notice: No Tavily API key. Using simulated fallback data.")
        result = "Simulated Fallback: Emmanuel Macron is the current President of France. He was born on December 21, 1977."
    else:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=api_key)
            # Find the most relevant context
            response = tavily.search(query=query, max_results=1)
            result = str(response)
        except Exception as e:
            result = f"Search failed: {e}"
            
    print(f"Result: {result}")
    return {
        "search_result": result,
        "messages": state["messages"] + [{"role": "system", "content": f"Search output: {result}"}]
    }

def finish_node(state: AgentState):
    print("\n--- FINISH NODE ---")
    query = state["query"]
    messages = state["messages"]
    
    prompt = f"""Provide the final answer to the user's initial query based on the collected information.
    Query: {query}
    History & Tool Data: {messages}
    
    Write a clear, concise final response."""
    
    response = llm.invoke(prompt)
    final_answer = response.content.strip()
    return {"final_answer": final_answer}

# =====================================================================
# 3. DEFINE THE ROUTER EDGE FUNCTION
# =====================================================================
def router_edge_func(state: AgentState) -> str:
    """Reads the current_step decision from state and directs the graph."""
    return state.get("current_step", "finish")

# =====================================================================
# 4. BUILD THE GRAPH
# =====================================================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("start", start_node)
workflow.add_node("plan", plan_node)
workflow.add_node("router", router_node)
workflow.add_node("calc", calculator_node)
workflow.add_node("search", search_node)
workflow.add_node("finish", finish_node)

# Define Core Connections
workflow.add_edge(START, "start")
workflow.add_edge("start", "plan")
workflow.add_edge("plan", "router")

# Define Conditional Edges
workflow.add_conditional_edges(
    "router",
    router_edge_func,
    {
        "calc": "calc",
        "search": "search",
        "finish": "finish"
    }
)

# Connect tools back to 'plan' to re-evaluate and check next steps
workflow.add_edge("calc", "plan")
workflow.add_edge("search", "plan")

# Connect final node to END
workflow.add_edge("finish", END)

# Compile with an in-memory checkpointer to persist state/debugging
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# =====================================================================
# 5. RUN TESTING SAMPLES
# =====================================================================
if __name__ == "__main__":
    # 3 Sample Prompts demonstrating different routing flows:
    # 1. Math computation only (Starts planning -> calc -> planning -> finish)
    # 2. Search query only (Starts planning -> search -> planning -> finish)
    # 3. Looping query requiring both tools sequentially
    sample_prompts = [
        # "What is 142857 multiplied by 6?",
        # "Who is the current President of France?",
        "Calculate the age of the current President of France multiplied by 2."
    ]
    
    for idx, prompt in enumerate(sample_prompts):
        print(f"\n{'='*50}")
        print(f"RUNNING PROMPT: '{prompt}'")
        print(f"{'='*50}")
        
        # We give each run a unique thread_id to ensure clean checkpoint states
        config = {"configurable": {"thread_id": f"test-thread-{idx}"}}
        
        initial_state = {"query": prompt}
        final_state = app.invoke(initial_state, config=config)
        
        print("\n🏆 [FINAL ANSWER]")
        print(final_state.get("final_answer"))