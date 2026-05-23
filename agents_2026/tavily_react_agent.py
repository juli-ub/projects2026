import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.agents import create_agent


'''This version uses LangChain's native agent library. You do not need to manually write the loop (answer->Search->Compare->correct), instead you define a web seach tool define the model, and write a system prompt that instructs the agent's interna reasoning loop (Reasoning + Acting, or React)
to perform the self-correctio autonomously.
the agent outputs a tool call request for the web search, with the parsed arguments, Tool Node: langchain intercepts this, uses your local web search function, tavily, and injects the output into the conversation state'''
# Load environment variables (ensure GOOGLE_API_KEY and TAVILY_API_KEY are set)
load_dotenv()

# 1. Define the Tavily Search Tool using the @tool decorator.
# The docstring acts as the instruction manual telling the agent when and how to use it.
@tool
def web_search(query: str) -> str:
    """Use this tool to search the web for real-time information and ground truth.
    It returns the top 3 web results containing titles, URLs, and text snippets.
    """
    search_tool = TavilySearch(max_results=3)
    try:
        results = search_tool.invoke({"query": query})
        
        # Format the results into a readable string for the agent
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
        
        formatted = []
        if isinstance(results, list):
            for i, r in enumerate(results, 1):
                formatted.append(f"Result {i}: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}\n")
            return "\n".join(formatted)
        return str(results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


# 2. Define the agent's system prompt.
# This explicitly instructs the agent to hypothesize, retrieve ground truth, and correct itself.
SYSTEM_PROMPT = """You are a highly analytical, self-correcting research assistant.

To answer the user's question accurately, you must follow this exact internal protocol:
1. Formulate an initial mental hypothesis or answer based on your existing knowledge (do not output this yet).
2. ALWAYS execute the 'web_search' tool to retrieve the live, real-time ground truth for the user's query.
3. Critically compare your initial hypothesis against the information retrieved from the web search.
4. If there is any discrepancy, outdated detail, or factual contradiction, correct your understanding immediately.
5. Provide a final response structured exactly as follows:
   
   - **My Initial Hypothesis**: (Briefly state what you initially assumed/thought before searching)
   - **Verification & Search Findings**: (Briefly explain what the search results revealed)
   - **Correction & Final Verified Answer**: (The corrected, highly accurate answer based on the live ground truth)
"""

def run_native_agent(query: str):
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Error: Ensure GOOGLE_API_KEY and TAVILY_API_KEY are configured in your .env file.")
        return

    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Compile the agent using LangChain's factory function
    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT
    )

    print(f"Sending Query to Agent: '{query}'\n")
    print("Agent is reasoning and executing tools...")
    
    # Run the agent graph
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    # Retrieve the final message from the conversation state history
    final_output = response["messages"][-1].content
    print("\n" + "="*40)
    print("AGENT RESPONSE:")
    print("="*40)
    print(final_output)

if __name__ == "__main__":
    # Test with a question that is prone to outdated information or hallucinations
    query = "Who is the current Prime Minister of the United Kingdom?"
    run_native_agent(query)