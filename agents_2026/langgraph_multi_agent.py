import os
import math
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# New LangChain v1 Standard Imports
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END, MessagesState

# 1. Load Environment Variables
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    print("Warning: Please ensure both GOOGLE_API_KEY and TAVILY_API_KEY are configured in your .env file.")

# 2. Initialize the Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# 3. Define Tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    Input must be a mathematical expression using numbers, operators (+, -, *, /), and standard python math functions.
    Examples: '12 * 34' or 'math.sqrt(144)' or '2**8'
    """
    try:
        # Evaluate with basic global constraints for minimal safety.
        # Note: In a production environment, avoid 'eval' and use library-based safe execution (e.g., SymPy).
        allowed_names = {
            "math": math,
            "abs": abs,
            "round": round,
            "pow": pow,
            "sum": sum,
        }
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error executing calculation: {str(e)}"

# Define the web search tool
search_tool = TavilySearchResults(max_results=3)

# 4. Create Specialist Agents (using LangChain's unified create_agent)
math_agent = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt="You are a mathematical expert. Solve math problems using the calculate tool. Be precise."
)

search_agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="You are a search expert. Find current information on the web using the tavily search tool. Keep findings concise."
)

# 5. Define Graph State (inherits 'messages' from LangGraph's MessagesState)
class AgentState(MessagesState):
    next: str

# 6. Define Supervisor Decision Schema
class RouterDecision(BaseModel):
    next_step: Literal["math_agent", "search_agent", "responder"] = Field(
        description="Select 'math_agent' for math/calculations, 'search_agent' for web searches, or 'responder' to formulate the final answer once all info is gathered."
    )
    reasoning: str = Field(description="Brief explanation of why this agent was chosen.")

# Bind the structured schema to the model to enforce routing decisions
supervisor_llm = llm.with_structured_output(RouterDecision)

# 7. Define Graph Nodes
def supervisor_node(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(content=(
        "You are the Supervisor. You coordinate a team of experts: 'math_agent' (for calculations) "
        "and 'search_agent' (for finding real-time facts/events/weather on the web). "
        "Your task is to review the conversation and decide which expert to call next, "
        "or call 'responder' to compile the final answer to the user once you have all the necessary information. "
        "If the user is asking a direct question that doesn't require tools, go straight to 'responder'."
    ))
    
    # Send the instruction along with the message history
    response = supervisor_llm.invoke([system_prompt] + list(messages))
    
    print(f"\n[Supervisor] Next Agent: {response.next_step}")
    print(f"[Supervisor] Reasoning: {response.reasoning}")
    
    return {"next": response.next_step}

def responder_node(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(content=(
        "You are the Responder. Your role is to look at the entire conversation, "
        "including all findings, calculations, and search results provided by your teammate experts, "
        "and draft a cohesive, well-structured final response for the user. "
        "Make sure to address all parts of the user's request."
    ))
    
    response = llm.invoke([system_prompt] + list(messages))
    return {"messages": [response]}

# 8. Build the Graph Workflow
workflow = StateGraph(AgentState)

# Add nodes to graph
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("math_agent", math_agent)
workflow.add_node("search_agent", search_agent)
workflow.add_node("responder", responder_node)

# Add entry point
workflow.add_edge(START, "supervisor")

# Define routing logic
def route_next(state: AgentState):
    return state["next"]

workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "math_agent": "math_agent",
        "search_agent": "search_agent",
        "responder": "responder"
    }
)

# Workers must report back to the supervisor once their job is done
workflow.add_edge("math_agent", "supervisor")
workflow.add_edge("search_agent", "supervisor")
workflow.add_edge("responder", END)

# Compile graph
app = workflow.compile()

# 9. Execution Helper
def run_prompt(user_prompt: str):
    print("\n" + "="*80)
    print(f"User Prompt: {user_prompt}")
    print("="*80)
    
    initial_state = {"messages": [HumanMessage(content=user_prompt)]}
    
    # Invoke the compiled graph
    result = app.invoke(initial_state)
    
    # The final message in state is our responder's output
    final_message = result["messages"][-1].content
    print("\n" + "-"*30 + " Final Answer " + "-"*30)
    print(final_message)
    print("="*80 + "\n")

if __name__ == "__main__":
    # --- Sample Prompt 1: Requires math only ---
    prompt_1 = "Calculate 15% of (540 + 360) and tell me the result."
    
    # --- Sample Prompt 2: Requires search only ---
    prompt_2 = "What are the latest updates regarding the Artemis II mission?"
    
    # --- Sample Prompt 3: Requires both search and math ---
    prompt_3 = "Find out how many employees Apple has in 2024 and then calculate what that number would be if they laid off 5% of their workforce."
    
    # Run the samples
    # run_prompt(prompt_1)
    # run_prompt(prompt_2)
    run_prompt(prompt_3)