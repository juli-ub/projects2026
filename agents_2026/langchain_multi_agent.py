import os
import math
from typing import Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

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

search_tool = TavilySearchResults(max_results=2)

# Dynamically map tools for retrieval during agent execution
tools_map = {
    calculate.name: calculate,
    search_tool.name: search_tool
}

# 4. Define Supervisor Decision Schema
class RouterDecision(BaseModel):
    next_step: Literal["math_agent", "search_agent", "responder"] = Field(
        description="Select 'math_agent' for math/calculations, 'search_agent' for web searches, or 'responder' if no more tools are needed."
    )
    reasoning: str = Field(description="Brief explanation of why this step was chosen.")

# Bind the structured schema to the model to enforce routing decisions
supervisor_llm = llm.with_structured_output(RouterDecision)

# 5. Core Specialist Function (Manages Tool Call lifecycle manually)
def run_specialist_agent(specialist_llm, system_message: str, user_prompt: str, chat_history: List[BaseMessage]) -> str:
    """Runs a specialist LLM bound to a specific tool. 
    Limits execution to at most 1 tool call to preserve free tier limit."""
    
    # Context given to the specialist includes the initial prompt and previous agent findings
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_prompt)
    ] + chat_history
    
    # 1st Invoke: Model decides if a tool is needed
    response = specialist_llm.invoke(messages)
    
    # If the model requests a tool call, we run it and feed the output back
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f" -> Specialist decides to run tool: {tool_name} with args: {tool_args}")
        
        # Invoke the correct tool
        if tool_name in tools_map:
            try:
                tool_output = tools_map[tool_name].invoke(tool_args)
            except Exception as e:
                tool_output = f"Error executing tool: {e}"
        else:
            tool_output = f"Error: Tool {tool_name} is not available."
            
        print(f" -> Tool output: {tool_output}")
        
        # Construct ToolMessage
        tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        
        # 2nd Invoke: Model digests the tool output and summarizes the finding
        final_summary = specialist_llm.invoke(messages + [response, tool_message])
        return final_summary.content
    else:
        # No tool call was requested by the specialist
        return response.content

# 6. Define the Orchestration Function (Pure Python Loop Control)
def run_multi_agent_system(user_prompt: str, max_supervisor_loops: int = 2):
    print("\n" + "="*80)
    print(f"User Prompt: {user_prompt}")
    print("="*80)
    
    chat_history: List[BaseMessage] = []
    called_agents = set()
    
    # Bind tools to specialists using LangChain's native .bind_tools API
    math_llm = llm.bind_tools([calculate])
    search_llm = llm.bind_tools([search_tool])
    
    for loop_idx in range(max_supervisor_loops):
        print(f"\n--- Orchestration Loop Iteration {loop_idx + 1}/{max_supervisor_loops} ---")
        
        system_prompt = SystemMessage(content=(
            "You are the Supervisor. You coordinate a team consisting of 'math_agent' (for calculations) "
            "and 'search_agent' (for finding real-time facts/events/weather on the web). "
            "Your task is to review the conversation and decide which expert to call next, "
            "or choose 'responder' if you have all the necessary information to answer the user's initial prompt."
        ))
        
        # The supervisor checks history and makes a decision
        decision = supervisor_llm.invoke([system_prompt, HumanMessage(content=user_prompt)] + chat_history)
        
        print(f"[Supervisor Decision] Next Step: {decision.next_step}")
        print(f"[Supervisor Reasoning] {decision.reasoning}")
        
        if decision.next_step == "responder":
            break
            
        if decision.next_step in called_agents:
            print(f"[Safeguard] {decision.next_step} was already executed. Skipping to responder to protect API usage.")
            break
            
        if decision.next_step == "math_agent":
            print("[System] Running Math Agent...")
            agent_output = run_specialist_agent(
                specialist_llm=math_llm,
                system_message="You are a mathematical expert. Solve math problems using the calculate tool. Be highly precise and brief.",
                user_prompt=user_prompt,
                chat_history=chat_history
            )
            print(f"[Math Agent Result] {agent_output}")
            chat_history.append(HumanMessage(content=f"Math Agent Output: {agent_output}"))
            called_agents.add("math_agent")
            
        elif decision.next_step == "search_agent":
            print("[System] Running Search Agent...")
            agent_output = run_specialist_agent(
                specialist_llm=search_llm,
                system_message="You are a search expert. Find current information on the web using the tavily search tool. Keep findings brief.",
                user_prompt=user_prompt,
                chat_history=chat_history
            )
            print(f"[Search Agent Result] {agent_output}")
            chat_history.append(HumanMessage(content=f"Search Agent Output: {agent_output}"))
            called_agents.add("search_agent")

    # 7. Final Response Compilation
    print("\n--- Compiling Final Answer ---")
    responder_prompt = [
        SystemMessage(content=(
            "You are the Responder. Review the user's initial prompt and the findings provided "
            "by the specialist agents in the conversation history. Draft a comprehensive, final response "
            "addressing the user's request."
        )),
        HumanMessage(content=user_prompt)
    ] + chat_history
    
    final_response = llm.invoke(responder_prompt)
    
    print("\n" + "-"*30 + " Final Answer " + "-"*30)
    print(final_response.content)
    print("="*80 + "\n")

if __name__ == "__main__":
    # Sample Prompt 1: Math execution
    prompt_1 = "Calculate 15% of (540 + 360) and tell me the result."
    
    # Sample Prompt 2: Search execution
    prompt_2 = "What are the latest updates regarding the Artemis II mission?"
    
    # Sample Prompt 3: Dual Execution (Search first, then calculate)
    prompt_3 = "Find out how many employees Apple has in 2024 and then calculate what that number would be if they laid off 5% of their workforce."
    
    # Execute the prompts sequentially
    # run_multi_agent_system(prompt_1)
    # run_multi_agent_system(prompt_2)
    run_multi_agent_system(prompt_3)