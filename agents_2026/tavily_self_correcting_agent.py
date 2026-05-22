import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# Load API keys from .env file
load_dotenv()

# Define the structured output format for our evaluator
class Evaluation(BaseModel):
    is_accurate: bool = Field(
        description="True if the LLM's answer aligns with the Tavily ground truth. False if there are contradictions, hallucinations, or major missing facts."
    )
    reason: str = Field(
        description="A brief explanation of why the answer is accurate, or what specific factual corrections are needed."
    )

def format_tavily_results(results: Any) -> str:
    """Safely parses and formats the output of TavilySearch tool into a readable string."""
    if isinstance(results, str):
        return results
    
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
        
    formatted = []
    if isinstance(results, list):
        for i, r in enumerate(results[:3], 1):  # Limit to top 3 results
            title = r.get("title", "No Title")
            url = r.get("url", "No URL")
            content = r.get("content", "No Content")
            formatted.append(f"Source {i}: {title}\nURL: {url}\nContent: {content}\n")
            
    return "\n".join(formatted) if formatted else str(results)


def get_initial_answer(query: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generates the initial response using the LLM's internal knowledge."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user query using your existing knowledge. Do not mention search engines or external tools."),
        ("human", "{query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return str(response.content)


def get_tavily_ground_truth(query: str) -> str:
    """Queries the Tavily Search API for the top 3 results."""
    search_tool = TavilySearch(max_results=3)
    try:
        results = search_tool.invoke({"query": query})
        return format_tavily_results(results)
    except Exception as e:
        return f"Error performing Tavily search: {str(e)}"


def evaluate_answer(query: str, initial_answer: str, tavily_context: str, llm: ChatGoogleGenerativeAI) -> Evaluation:
    """Compares the initial answer against Tavily search results using a structured output chain."""
    structured_llm = llm.with_structured_output(Evaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a factual validation assistant. Your job is to rigorously compare an LLM response against web search ground truth."),
        ("human", (
            "User Query: {query}\n\n"
            "Initial LLM Answer:\n{initial_answer}\n\n"
            "Tavily Search Ground Truth (Top 3 Results):\n{tavily_context}\n\n"
            "Identify if there are any factual discrepancies, outdated statements, or significant omissions "
            "in the Initial LLM Answer compared to the Tavily Ground Truth."
        ))
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({
        "query": query,
        "initial_answer": initial_answer,
        "tavily_context": tavily_context
    })


def get_corrected_answer(query: str, initial_answer: str, tavily_context: str, reason: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generates a new corrected answer using the Tavily results and correction guidelines."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a self-correcting assistant. Use the provided search results to correct and rewrite your previous response."),
        ("human", (
            "User Query: {query}\n\n"
            "Your Previous Answer:\n{initial_answer}\n\n"
            "Feedback from Validator:\n{reason}\n\n"
            "Top 3 Tavily Search Results (Ground Truth):\n{tavily_context}\n\n"
            "Please output a corrected, comprehensive response. Focus specifically on addressing the discrepancies noted in the feedback."
        ))
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "query": query,
        "initial_answer": initial_answer,
        "reason": reason,
        "tavily_context": tavily_context
    })
    return str(response.content)


def run_agent(query: str):
    """Orchestrates the self-correcting loop."""
    # Ensure API keys are present
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Error: Please set GOOGLE_API_KEY and TAVILY_API_KEY in your environment or .env file.")
        return

    # Initialize Gemini model (Using gemini-1.5-flash for rapid, cost-effective performance)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    print(f"\n--- Processing Query: '{query}' ---")
    
    # Step 1: Get Initial Answer from the LLM
    print("\n[1/4] Generating initial answer...")
    initial_answer = get_initial_answer(query, llm)
    print(f"Initial LLM Answer:\n{initial_answer}")

    # Step 2: Fetch ground truth via Tavily
    print("\n[2/4] Fetching search results from Tavily...")
    tavily_context = get_tavily_ground_truth(query)

    # Step 3: Evaluate the initial answer
    print("\n[3/4] Evaluating initial answer against ground truth...")
    evaluation = evaluate_answer(query, initial_answer, tavily_context, llm)
    print(f"Is Accurate? {evaluation.is_accurate}")
    print(f"Evaluation Reason: {evaluation.reason}")

    # Step 4: Correct the answer if the evaluator flags it
    if not evaluation.is_accurate:
        print("\n[4/4] Inaccuracy detected. Generating corrected answer...")
        corrected_answer = get_corrected_answer(
            query=query,
            initial_answer=initial_answer,
            tavily_context=tavily_context,
            reason=evaluation.reason,
            llm=llm
        )
        print(f"\nFinal Corrected Answer:\n{corrected_answer}")
    else:
        print("\n[4/4] Initial answer was determined to be accurate. No correction needed.")
        print(f"\nFinal Answer:\n{initial_answer}")


if __name__ == "__main__":
    # Test query (e.g., querying recent events to test the search verification)
    test_query = "Who won the men's singles title at the most recent Wimbledon championships?"
    run_agent(test_query)