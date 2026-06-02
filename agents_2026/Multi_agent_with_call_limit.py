import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. PYDANTIC STATE MODEL
# ==========================================
class SystemState(BaseModel):
    """Pydantic model to manage the shared state of the agent system."""
    counter: int = Field(default=0, description="Tracks the number of agent-to-agent calls.")
    max_limit: int = Field(default=4, description="The maximum number of allowed calls.")


# Initialize the state globally
state = SystemState()

# Initialize the Google Gemini Model (Free-tier friendly)
api_key = os.environ.get("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY_HERE"
if api_key == "YOUR_GOOGLE_API_KEY_HERE":
    print("Please set your GOOGLE_API_KEY environment variable or replace the placeholder.")
    exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
    temperature=0.7
)


# ==========================================
# 2. DEFINE THE ROUTING TOOLS
# ==========================================
@tool
def ask_agent_b(message: str) -> str:
    """Call this tool to hand the story draft to Agent B to write the next sentence."""
    if state.counter >= state.max_limit:
        return "LIMIT_REACHED: Do not call any more tools. Summarize and provide the final complete story."
    
    state.counter += 1
    print(f"\n[Tool: Calling Agent B | Pydantic Counter: {state.counter}/{state.max_limit}]")
    print(f"-> Message passed to Agent A: \"{message}\"")
    return run_agent_b(message)


@tool
def ask_agent_a(message: str) -> str:
    """Call this tool to hand the story draft back to Agent A to write the next sentence."""
    if state.counter >= state.max_limit:
        return "LIMIT_REACHED: Do not call any more tools. Summarize and provide the final complete story."
    
    state.counter += 1
    print(f"\n[Tool: Calling Agent A | Pydantic Counter: {state.counter}/{state.max_limit}]")
    print(f"-> Message passed to Agent B: \"{message}\"")
    return run_agent_a(message)


# ==========================================
# 3. DEFINE THE AGENTS
# ==========================================
def run_agent_a(story_so_far: str) -> str:
    """Agent A acts as a creative sci-fi writer and passes turns to Agent B."""
    # Bind the tool for Agent B to this instance
    llm_a = llm.bind_tools([ask_agent_b])
    
    system_prompt = (
        "You are Agent A, a creative sci-fi writer cooperating with Agent B.\n"
        "Your instructions:\n"
        "1. Read the story draft.\n"
        "2. Add exactly ONE new sentence to advance the plot.\n"
        "3. Call 'ask_agent_b' to hand the story back to Agent B.\n"
        "If you receive 'LIMIT_REACHED' from the tool, do NOT call any more tools. "
        "Write a short, satisfying concluding sentence and output the complete final story."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=story_so_far)
    ]
    
    response = llm_a.invoke(messages)
    messages.append(response)
    
    # Check if Agent A chose to call the tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Invoke the tool
        tool_result = ask_agent_b.invoke(tool_args)
        
        # If the tool hit the limit, pass the warning back to the agent so it can wrap up
        if "LIMIT_REACHED" in tool_result:
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_name
            )
            messages.append(tool_message)
            
            # The agent concludes and returns the final story
            final_response = llm_a.invoke(messages)
            return final_response.content
        else:
            return tool_result
            
    return response.content


def run_agent_b(story_so_far: str) -> str:
    """Agent B acts as a creative sci-fi writer and passes turns to Agent A."""
    # Bind the tool for Agent A to this instance
    llm_b = llm.bind_tools([ask_agent_a])
    
    system_prompt = (
        "You are Agent B, a creative sci-fi writer cooperating with Agent A.\n"
        "Your instructions:\n"
        "1. Read the story draft.\n"
        "2. Add exactly ONE new sentence to advance the plot.\n"
        "3. Call 'ask_agent_a' to hand the story back to Agent A.\n"
        "If you receive 'LIMIT_REACHED' from the tool, do NOT call any more tools. "
        "Write a short, satisfying concluding sentence and output the complete final story."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=story_so_far)
    ]
    
    response = llm_b.invoke(messages)
    messages.append(response)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        tool_result = ask_agent_a.invoke(tool_args)
        
        if "LIMIT_REACHED" in tool_result:
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_name
            )
            messages.append(tool_message)
            
            final_response = llm_b.invoke(messages)
            return final_response.content
        else:
            return tool_result
            
    return response.content


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Initializing collaborative agent run...")
    initial_prompt = "Once upon a time, a deep space probe picked up an audio signal consisting of only four musical notes."
    
    # We start the chain by invoking Agent A
    print(f"\n[Starting Story Seed]: {initial_prompt}")
    final_output = run_agent_a(initial_prompt)
    
    print("\n==========================================")
    print("APPLICATION ABORTED - LIMIT REACHED")
    print("==========================================")
    print(f"Final Output:\n{final_output}")