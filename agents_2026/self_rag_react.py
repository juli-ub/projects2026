import os
import sys
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent

# Load API key from .env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    sys.exit(1)

# Helper: Create a sample text file if it doesn't exist
def create_sample_file():
    filename = "knowledge.txt"
    if not os.path.exists(filename):
        content = """
        The Apollo 11 mission landed the first humans on the Moon. Commander Neil Armstrong 
        and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. 
        Neil Armstrong became the first person to step onto the lunar surface six hours and 39 minutes 
        later on July 21. Buzz Aldrin joined him 19 minutes later.

        The Great Wall of China is a series of fortifications that were built across the historical northern 
        borders of ancient Chinese states and Imperial China as protection against various nomadic groups. 
        Several walls were built from as early as the 7th century BC.

        Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy 
        that, through cellular respiration, can later be released to fuel the organisms' activities.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Created a dummy context file: {filename}")


class SelfRAGWorkflow:
    """Encapsulates the self-corrective RAG loop logic."""
    def __init__(self, filepath="knowledge.txt"):
        print("[Self-RAG System] Initializing Models & Vector Store...")
        
        # Initialize Google Gemini models
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # Load and chunk text file
        self.vectorstore = self._build_vector_store(filepath)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def _build_vector_store(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = text_splitter.create_documents([text])
        
        # Embed and store inside a local FAISS index
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        return vectorstore

    def grade_document_relevance(self, query: str, document_content: str) -> bool:
        prompt = f"""You are an objective grader assessing the relevance of a retrieved document to a user question.
        
        Retrieved Document:
        {document_content}
        
        User Question:
        {query}
        
        Does this document contain information relevant to answering the question? Answer with only 'yes' or 'no'."""
        
        response = self.llm.invoke(prompt)
        decision = response.content.strip().lower()
        return "yes" in decision

    def rewrite_query(self, original_query: str) -> str:
        prompt = f"""You are an AI assistant tasked with improving a search query for a vector database. 
        Your goal is to make it more semantically rich.
        Original query: {original_query}
        Provide only the optimized search query text and nothing else."""
        
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. Use the provided context below to answer the user's question. 
        If you do not know the answer or if it's not present in the context, say 'I cannot find the answer in the provided documents.' 
        Do not make up facts.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def check_hallucination(self, context: str, generation: str) -> bool:
        prompt = f"""You are a strict factual grader. Check if the generated answer relies on any facts NOT present in the context.
        
        Context:
        {context}
        
        Generated Answer:
        {generation}
        
        Is the generated answer completely grounded in and supported by the provided context? Answer with only 'yes' or 'no'."""
        
        response = self.llm.invoke(prompt)
        decision = response.content.strip().lower()
        return "yes" in decision

    def run(self, query: str) -> str:
        """Executes the Self-RAG loop and returns the validated output."""
        print(f"\n[Self-RAG WorkFlow] Processing Inner Query: '{query}'")
        
        # Step 1: Retrieval
        print("  -> Retrieving documents...")
        retrieved_docs = self.retriever.invoke(query)
        
        # Step 2: Document Relevance Grading
        print("  -> Grading document relevance...")
        relevant_chunks = []
        for i, doc in enumerate(retrieved_docs):
            is_relevant = self.grade_document_relevance(query, doc.page_content)
            print(f"     - Chunk {i+1}: {'RELEVANT' if is_relevant else 'IRRELEVANT'}")
            if is_relevant:
                relevant_chunks.append(doc.page_content)
                
        # Step 3: Fallback & Query Rewriting
        if not relevant_chunks:
            print("  -> No relevant documents found. Rewriting query...")
            new_query = self.rewrite_query(query)
            print(f"     - New Query: '{new_query}'")
            print("     - Re-retrieving documents...")
            retrieved_docs = self.retriever.invoke(new_query)
            
            for i, doc in enumerate(retrieved_docs):
                is_relevant = self.grade_document_relevance(new_query, doc.page_content)
                print(f"       - Chunk {i+1}: {'RELEVANT' if is_relevant else 'IRRELEVANT'}")
                if is_relevant:
                    relevant_chunks.append(doc.page_content)
                    
            if not relevant_chunks:
                return "Workflow failed to find relevant information in local records."

        # Step 4: Generation
        context_str = "\n\n".join(relevant_chunks)
        print("  -> Generating draft answer...")
        answer = self.generate_answer(query, context_str)
        
        # Step 5: Hallucination Verification
        print("  -> Running hallucination check...")
        is_grounded = self.check_hallucination(context_str, answer)
        
        if is_grounded:
            print("  -> Passed factual validation.")
            return answer
        else:
            print("  -> Warning: Draft contained ungrounded information. Regenerating strictly...")
            strict_prompt = f"Regenerate the answer to '{query}' strictly using only this text: {context_str}. Do not extrapolate."
            strict_answer = self.llm.invoke(strict_prompt).content.strip()
            return strict_answer


# --- Set up the Tool and Main Agent ---

create_sample_file()

# 1. Initialize the workflow engine
rag_workflow = SelfRAGWorkflow("knowledge.txt")

# 2. Define the tool with the @tool decorator
# The docstring instructs the main agent on when and how to call this tool
@tool
def query_local_records(query: str) -> str:
    """
    Search the local knowledge base to retrieve verified factual data. 
    Use this tool whenever the user asks questions about historical facts, 
    the Apollo 11 moon landing, the Great Wall of China, or photosynthesis.
    
    Args:
        query (str): The search phrase or question to look up.
    """
    return rag_workflow.run(query)

# 3. Create the Main Chat Model for the Agent
main_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. Construct the agent using create_agent (the modern langchain standard)
# Under the hood, this compiles into a LangGraph state machine
agent = create_agent(
    model=main_llm,
    tools=[query_local_records],
    system_prompt=(
        "You are a factual research assistant. "
        "Use the 'query_local_records' tool to look up information from our database "
        "when asked about the Apollo 11 mission, the Great Wall of China, or photosynthesis."
    )
)


# --- Execution Block ---
if __name__ == "__main__":
    print("\n=== Launching LangChain Agentic Loop ===")
    
    # Define a high-level query for the agent
    user_query = "Can you check who was the Lunar Module Pilot on Apollo 11 according to our database?"
    print(f"\n[User Query]: {user_query}")
    
    # Standard format for invoking create_agent: passing a list of message dicts
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_query}
        ]
    })
    
    # Extract the final agent output from the message history
    final_answer = response["messages"][-1].content
    print(f"\n[Final Agent Answer]:\n{final_answer}\n")