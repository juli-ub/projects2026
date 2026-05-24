import os
import math
from typing import Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# LangChain v1 Agent Creation Standard
from langchain.agents import create_agent

# 1. Load Environment Variables
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    print("Warning: Please ensure both GOOGLE_API_KEY and TAVILY_API_KEY are configured in your .env file.")

# 2. Initialize the Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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

# 4. Create Specialist Agents with LangChain's Up-to-Date 'create_agent' API
# Note: create_agent handles the tool-calling ReAct loop internally.
math_agent = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt="You are a mathematical expert. Solve math problems using the calculate tool. Be highly precise and brief."
)

search_agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="You are a search expert. Find current information on the web using the tavily search tool. Keep findings brief."
)

# 5. Define Supervisor Decision Schema
class RouterDecision(BaseModel):
    next_step: Literal["math_agent", "search_agent", "responder"] = Field(
        description="Select 'math_agent' for math/calculations, 'search_agent' for web searches, or 'responder' if no more tools are needed."
    )
    reasoning: str = Field(description="Brief explanation of why this step was chosen.")

supervisor_llm = llm.with_structured_output(RouterDecision)

# 6. Define the Orchestration Function (Pure Python Loop)
def run_multi_agent_system(user_prompt: str, max_supervisor_loops: int = 3):
    print("\n" + "="*80)
    print(f"User Prompt: {user_prompt}")
    print("="*80)
    
    # Track findings accumulated from the agents
    chat_history: List[BaseMessage] = []
    
    # We allow up to 3 loops to ensure that both tools can be called in sequence,
    # and the supervisor can verify the final state before responding.
    for loop_idx in range(max_supervisor_loops):
        print(f"\n--- Orchestration Loop Iteration {loop_idx + 1}/{max_supervisor_loops} ---")
        
        system_prompt = SystemMessage(content=(
            "You are the Supervisor. You coordinate a team consisting of 'math_agent' (for calculations) "
            "and 'search_agent' (for finding real-time facts/events/weather on the web). "
            "Your task is to review the conversation and decide which expert to call next, "
            "or choose 'responder' if you have all the necessary information to answer the user's initial prompt. "
            "Do not call an agent if its task has already been fully completed in the conversation history."
        ))
        
        # Prepare context for the supervisor
        supervisor_input = [system_prompt, HumanMessage(content=user_prompt)] + chat_history
        
        # Call supervisor to make a structured decision
        decision = supervisor_llm.invoke(supervisor_input)
        
        print(f"[Supervisor Decision] Next Step: {decision.next_step}")
        print(f"[Supervisor Reasoning] {decision.reasoning}")
        
        if decision.next_step == "responder":
            break
            
        if decision.next_step == "math_agent":
            print("[System] Running Math Agent...")
            # We feed the initial prompt + findings history into the specialist agent
            agent_input_messages = [HumanMessage(content=user_prompt)] + chat_history
            
            # create_agent outputs a dictionary containing a 'messages' list with all steps included
            result = math_agent.invoke({"messages": agent_input_messages})
            
            # Extract the agent's final summary message
            final_output = result["messages"][-1].content
            print(f"[Math Agent Result] {final_output}")
            
            chat_history.append(HumanMessage(content=f"Math Agent Output: {final_output}"))
            
        elif decision.next_step == "search_agent":
            print("[System] Running Search Agent...")
            agent_input_messages = [HumanMessage(content=user_prompt)] + chat_history
            
            result = search_agent.invoke({"messages": agent_input_messages})
            
            final_output = result["messages"][-1].content
            print(f"[Search Agent Result] {final_output}")
            
            chat_history.append(HumanMessage(content=f"Search Agent Output: {final_output}"))

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
    # --- Sample Prompts ---
    
    # Prompt 1: Requires math execution
    prompt_1 = "Calculate 15% of (540 + 360) and tell me the result."
    
    # Prompt 2: Requires search execution
    prompt_2 = "What are the latest updates regarding the Artemis II mission?"
    
    # Prompt 3: Requires both search and math execution
    prompt_3 = "Find out how many employees Apple has in 2024 and then calculate what that number would be if they laid off 5% of their workforce."
    
    # Run the samples
    # run_multi_agent_system(prompt_1)
    # run_multi_agent_system(prompt_2)
    run_multi_agent_system(prompt_3)