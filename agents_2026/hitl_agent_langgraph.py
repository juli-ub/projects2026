import os
import uuid
import getpass
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================================
# 1. API Key Setup Check
# =====================================================================
if "GOOGLE_API_KEY" not in os.environ:
    # If the environment variable isn't set, prompt for it in the terminal
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize the Gemini model
# We use gemini-1.5-flash for fast and cost-efficient responses
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

# =====================================================================
# 2. Define the Shared State
# =====================================================================
class AgentState(TypedDict):
    topic: str
    draft: str
    feedback: Optional[str]
    final_output: Optional[str]

# =====================================================================
# 3. Define the Graph Nodes (Functions connected to Google Gen AI)
# =====================================================================

def draft_node(state: AgentState):
    """Generates the initial draft using Google Gen AI."""
    print("\n--- [Node: Drafting Content via Gemini] ---")
    topic = state.get("topic", "General AI Agents")
    
    prompt = f"Write a short, engaging introductory paragraph (maximum 3 sentences) about the following topic: {topic}"
    response = llm.invoke(prompt)
    
    return {"draft": response.content.strip()}


def human_review_node(state: AgentState):
    """Pauses the graph and waits for user input."""
    print("\n--- [Node: Human Review Node] ---")
    
    # Pauses and returns the user payload once resumed
    user_response = interrupt({
        "prompt": "Please review this draft. Enter 'approve' to proceed, or write feedback to request modifications.",
        "draft_to_review": state["draft"]
    })
    
    return {"feedback": user_response}


def route_feedback(state: AgentState) -> str:
    """Conditional router determining if we edit again or finish."""
    feedback = state.get("feedback", "")
    if isinstance(feedback, str) and feedback.strip().lower() == "approve":
        return "publish"
    else:
        return "redraft"


def redraft_node(state: AgentState):
    """Modifies the draft using Google Gen AI based on your feedback."""
    print("\n--- [Node: Modifying Draft via Gemini] ---")
    feedback = state.get("feedback", "")
    current_draft = state.get("draft", "")
    
    prompt = f"""
    You are an expert editor. Please rewrite the current draft based strictly on the provided feedback.
    
    Current Draft:
    "{current_draft}"
    
    Feedback:
    "{feedback}"
    
    Return only the revised draft, keeping it brief and professional.
    """
    response = llm.invoke(prompt)
    
    return {"draft": response.content.strip()}


def publish_node(state: AgentState):
    """Outputs the finalized response."""
    print("\n--- [Node: Publishing Content] ---")
    return {"final_output": f"PUBLISHED RESPONSE:\n{state['draft']}"}

# =====================================================================
# 4. Build and Compile the Graph
# =====================================================================
builder = StateGraph(AgentState)

builder.add_node("draft_node", draft_node)
builder.add_node("human_review_node", human_review_node)
builder.add_node("redraft_node", redraft_node)
builder.add_node("publish_node", publish_node)

builder.add_edge(START, "draft_node")
builder.add_edge("draft_node", "human_review_node")

# Conditional routing
builder.add_conditional_edges(
    "human_review_node",
    route_feedback,
    {
        "publish": "publish_node",
        "redraft": "redraft_node"
    }
)

builder.add_edge("redraft_node", "human_review_node")
builder.add_edge("publish_node", END)

# In-memory checkpointing saves state during interrupts
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# =====================================================================
# 5. Runtime Loop
# =====================================================================
def run_agent():
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("Starting agent workflow...")
    initial_input = {"topic": "The future of Human-in-the-loop AI workflows"}
    
    # Run initial nodes
    for event in graph.stream(initial_input, config=thread_config, stream_mode="updates"):
        pass
        
    state = graph.get_state(thread_config)
    
    while state.next:
        if state.interrupts:
            active_interrupt = state.interrupts[0]
            
            print("\n" + "="*50)
            print("HUMAN INTERVENTION REQUIRED")
            print("="*50)
            print(f"Instruction: {active_interrupt.value['prompt']}")
            print(f"Draft:       {active_interrupt.value['draft_to_review']}")
            print("="*50)
            
            # Request local console input
            user_input = input("\nYour Response: ")
            
            print("\nResuming workflow with your feedback...")
            # Resume graph execution passing the user input
            for event in graph.stream(Command(resume=user_input), config=thread_config, stream_mode="updates"):
                pass
                
            state = graph.get_state(thread_config)
        else:
            break
            
    print("\nWorkflow Completed!")
    final_state = graph.get_state(thread_config)
    print("\n" + "#"*50)
    print(final_state.values.get("final_output"))
    print("#"*50)

if __name__ == "__main__":
    run_agent()