import os
import sys
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

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

class SelfRAGAgent:
    def __init__(self, filepath="knowledge.txt"):
        print("[System] Initializing Models & Vector Store...")
        
        # Initialize Google Gemini models
        # gemini-embedding-001 is the standard text embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # gemini-2.5-flash is ideal for fast, cost-effective reasoning
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
        """Determines if a retrieved chunk is relevant to the user's question."""
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
        """Optimizes the query for better semantic retrieval if initial attempts fail."""
        prompt = f"""You are an AI assistant tasked with improving a search query for a vector database. 
        Your goal is to make it more semantically rich.
        Original query: {original_query}
        Provide only the optimized search query text and nothing else."""
        
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def generate_answer(self, query: str, context: str) -> str:
        """Synthesizes the final answer using only the provided context."""
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
        """Grades whether the generated answer is strictly grounded in the retrieved context (no hallucination)."""
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
        """Executes the Self-RAG agent workflow loop."""
        print(f"\n--- Processing Query: '{query}' ---")
        
        # Step 1: Retrieval
        print("[Agent Step 1] Retrieving documents from FAISS...")
        retrieved_docs = self.retriever.invoke(query)
        
        # Step 2: Document Relevance Grading
        print("[Agent Step 2] Grading document relevance...")
        relevant_chunks = []
        for i, doc in enumerate(retrieved_docs):
            is_relevant = self.grade_document_relevance(query, doc.page_content)
            print(f"  - Chunk {i+1}: {'RELEVANT' if is_relevant else 'IRRELEVANT'}")
            if is_relevant:
                relevant_chunks.append(doc.page_content)
                
        # Step 3: Fallback & Query Rewriting
        if not relevant_chunks:
            print("[Agent Step 3 - Loop] No relevant documents found. Rewriting query...")
            new_query = self.rewrite_query(query)
            print(f"  - New Query: '{new_query}'")
            print("  - Re-retrieving documents...")
            retrieved_docs = self.retriever.invoke(new_query)
            
            for i, doc in enumerate(retrieved_docs):
                is_relevant = self.grade_document_relevance(new_query, doc.page_content)
                print(f"    - Chunk {i+1}: {'RELEVANT' if is_relevant else 'IRRELEVANT'}")
                if is_relevant:
                    relevant_chunks.append(doc.page_content)
                    
            if not relevant_chunks:
                return "Agent failed to find relevant information in the local files to answer your query."

        # Step 4: Generation
        context_str = "\n\n".join(relevant_chunks)
        print("[Agent Step 4] Generating draft response...")
        answer = self.generate_answer(query, context_str)
        
        # Step 5: Hallucination Verification
        print("[Agent Step 5] Running hallucination check...")
        is_grounded = self.check_hallucination(context_str, answer)
        
        if is_grounded:
            print("  - Response passed validation. No hallucinations detected.")
            return answer
        else:
            print("  - Warning: Draft contained information outside the context. Attempting strict regeneration...")
            strict_prompt = f"Regenerate the answer to '{query}' strictly using only this text: {context_str}. Do not extrapolate."
            strict_answer = self.llm.invoke(strict_prompt).content.strip()
            return strict_answer

# Execution Block
if __name__ == "__main__":
    create_sample_file()
    
    # Initialize the agent
    agent = SelfRAGAgent("knowledge.txt")
    
    # Scenario A: Information is directly available
    query_a = "Who was the Lunar Module Pilot on Apollo 11?"
    result_a = agent.run(query_a)
    print(f"\n[Final Agent Answer]:\n{result_a}\n")
    
    # Scenario B: Requires Query Rewriting (using synonyms or concepts not directly in the document name)
    # query_b = "Tell me about solar-powered conversion in foliage." 
    # # (Rewriter will likely map 'foliage' and 'solar' to 'Photosynthesis')
    # result_b = agent.run(query_b)
    # print(f"\n[Final Agent Answer]:\n{result_b}\n")