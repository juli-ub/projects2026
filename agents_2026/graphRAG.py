import os
import difflib
from typing import List
import networkx as nx

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
# 1. Load Environment Variables
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("Warning: Please ensure GOOGLE_API_KEY is configured in your .env file.")
# ==========================================
# 1. INITIALIZE GEMINI LLM
# ==========================================
# If you didn't set the environment variable, uncomment the line below:
# os.environ["GOOGLE_API_KEY"] = "AIzaSy..." 

# Using gemini-1.5-flash for fast execution and robust JSON output parsing
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# ==========================================
# 2. DEFINE THE STRUCTURE FOR GRAPH EXTRACTION
# ==========================================
class Relationship(BaseModel):
    source: str = Field(description="The starting concept or system (e.g., 'Auth Service')")
    relation: str = Field(description="The relationship link (e.g., 'depends on', 'manages', 'writes to')")
    target: str = Field(description="The ending concept or system (e.g., 'Payment Service', 'Auth DB')")

class KnowledgeGraph(BaseModel):
    relationships: List[Relationship] = Field(description="List of all extracted relationships")

parser = JsonOutputParser(pydantic_object=KnowledgeGraph)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise technical data extractor. "
        "Extract relationships from the text as a list of triplets (source, relation, target).\n"
        "Keep names clean, consistent, and short (e.g., use 'Auth Service' instead of 'The main authentication system').\n"
        "Format your output strictly as JSON matching this schema:\n{format_instructions}"
    )),
    ("human", "Text to extract from:\n{text}")
])

extraction_chain = extraction_prompt | llm | parser

# ==========================================
# 3. SAMPLE DOCUMENTATION (INPUT DATA)
# ==========================================
documents = [
    "The Auth Service handles user login and session management. It reads and writes data to the Auth DB.",
    "The Payment Service processes transactions. It depends on the Auth Service to validate active user sessions.",
    "The Billing DB is owned and used exclusively by the Payment Service.",
    "The Notification Service sends confirmation emails to users. It depends on the Payment Service to trigger events.",
    "Team Alpha owns and maintains the Auth Service.",
    "Team Beta is responsible for the Payment Service and the Billing DB.",
    "Team Gamma manages the Notification Service and handles customer outreach."
]

# ==========================================
# 4. BUILD THE LOCAL KNOWLEDGE GRAPH
# ==========================================
G = nx.Graph()

print("Analyzing documentation chunks and building local Knowledge Graph...")
for i, doc in enumerate(documents):
    try:
        # Request Gemini to structure the raw text into relationships
        result = extraction_chain.invoke({
            "text": doc, 
            "format_instructions": parser.get_format_instructions()
        })
        for rel in result.get("relationships", []):
            source = rel["source"].strip()
            target = rel["target"].strip()
            relation = rel["relation"].strip()
            
            # Map elements into our local NetworkX graph structure
            G.add_edge(source, target, relation=relation)
            print(f" -> Extracted: ({source}) --[{relation}]--> ({target})")
    except Exception as e:
        print(f"Skipped document chunk {i} due to formatting differences: {e}")

print("\nKnowledge Graph constructed successfully!")
print(f"Total Nodes: {G.number_of_nodes()} | Total Edges: {G.number_of_edges()}\n")

# ==========================================
# 5. RETRIEVAL: ENTITY EXTRACTION & N-HOP TRAVERSAL
# ==========================================
class FocusEntity(BaseModel):
    entity: str = Field(description="The primary system, service, database, or team mentioned in the query.")

entity_parser = JsonOutputParser(pydantic_object=FocusEntity)
entity_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Identify the primary technical system, service, database, or team the user is asking about.\n"
        "Output your answer in this JSON format:\n{format_instructions}"
    )),
    ("human", "User Query: {query}")
])

entity_chain = entity_prompt | llm | entity_parser

def find_closest_node(graph, target):
    """Finds the closest node name in the graph to account for minor spelling variations."""
    nodes = list(graph.nodes())
    closest = difflib.get_close_matches(target, nodes, n=1, cutoff=0.3)
    return closest[0] if closest else None

def get_subgraph_context(graph, start_node, hops=3):
    """Retrieves all connected edges up to N-hops away to build context for multi-hop queries."""
    matched_node = find_closest_node(graph, start_node)
    if not matched_node:
        return f"System entity '{start_node}' not found in the local knowledge base."
    
    sub_nodes = {matched_node}
    for _ in range(hops):
        next_nodes = set()
        for node in sub_nodes:
            next_nodes.update(graph.neighbors(node))
        sub_nodes.update(next_nodes)
        
    subgraph = graph.subgraph(sub_nodes)
    
    context_lines = [
        f"Focus Entity matched in Graph: '{matched_node}'",
        "Retrieved topology pathways:"
    ]
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'is connected to')
        context_lines.append(f" - [{u}] --({relation})--> [{v}]")
        
    return "\n".join(context_lines)

# ==========================================
# 6. GENERATION: GRAPH-AUGMENTED RESPONSE
# ==========================================
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an IT Systems Analyst.\n"
        "Use the retrieved system topology connections to answer the user's question.\n"
        "Trace relationships step-by-step to show downstream dependencies and impacts.\n\n"
        "--- RETRIEVED TOPOLOGY CONTEXT ---\n"
        "{context}\n"
        "------------------------------------"
    )),
    ("human", "User Question: {query}")
])

rag_chain = rag_prompt | llm

# ==========================================
# 7. EXECUTION
# ==========================================
user_query = "If we make breaking changes to the Auth Service, which downstream services are affected, and which teams do we need to alert?"
print(f"User Query: '{user_query}'\n")

# Step A: Identify the core entity in the query
extraction_result = entity_chain.invoke({
    "query": user_query,
    "format_instructions": entity_parser.get_format_instructions()
})
extracted_entity = extraction_result.get("entity", "")

# Step B: Perform graph retrieval (traverse 3 hops to capture indirect dependents)
retrieved_context = get_subgraph_context(G, extracted_entity, hops=3)
print("--- RETRIEVED GRAPH CONTEXT ---")
print(retrieved_context)
print("-------------------------------\n")

# Step C: Generate final system impact analysis
print("Generating analysis...")
final_response = rag_chain.invoke({
    "query": user_query,
    "context": retrieved_context
})

print("\n--- FINAL GRAPH RAG ANALYSIS ---")
print(final_response.content)