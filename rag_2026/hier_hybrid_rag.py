import os
import math
from collections import Counter

# Friendly imports check to guide setup
try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError:
    print("\n[!] Missing required packages. Please run the following command in your terminal:")
    print("pip install numpy sentence-transformers transformers torch\n")
    exit(1)


# =====================================================================
# 1. Self-Contained BM25 Keyword Search
# =====================================================================
class SimpleBM25:
    """A lightweight, dependency-free BM25 implementation for lexical keyword search."""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = (
            sum(len(doc.split()) for doc in corpus) / self.corpus_size 
            if self.corpus_size > 0 else 1
        )
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.tf = []
        
        for doc in corpus:
            words = doc.lower().split()
            self.doc_lens.append(len(words))
            self.tf.append(Counter(words))
            for word in set(words):
                self.doc_freqs[word] += 1
                
        self.idf = {}
        for word, freq in self.doc_freqs.items():
            # Standard smoothed IDF formula
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_score(self, query, doc_index):
        query_words = query.lower().split()
        score = 0.0
        tf = self.tf[doc_index]
        doc_len = self.doc_lens[doc_index]
        
        for word in query_words:
            if word in tf:
                word_tf = tf[word]
                idf = self.idf.get(word, 0)
                numerator = word_tf * (self.k1 + 1)
                denominator = word_tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
        return score


# =====================================================================
# 2. Hybrid Search Scoring
# =====================================================================
def normalize_scores(scores):
    """Normalize score arrays to [0, 1] for safe combination."""
    scores = np.array(scores)
    min_s = scores.min()
    max_s = scores.max()
    if max_s == min_s:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)

def compute_hybrid_scores(query, documents, dense_model, alpha=0.5):
    """Combines Dense Cosine Similarity and BM25 scores."""
    if not documents:
        return []
    
    # A. Semantic Search (Dense)
    query_emb = dense_model.encode(query, convert_to_tensor=True).cpu().numpy()
    doc_embs = dense_model.encode(documents, convert_to_tensor=True).cpu().numpy()
    
    # Compute Cosine Similarity
    dot_products = np.dot(doc_embs, query_emb)
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    doc_norms[doc_norms == 0] = 1e-9  # Avoid division by zero
    semantic_scores = dot_products / (query_norm * doc_norms)
    
    # B. Keyword Search (BM25)
    bm25 = SimpleBM25(documents)
    bm25_scores = np.array([bm25.get_score(query, i) for i in range(len(documents))])
    
    # C. Min-Max Normalization and Fusion
    norm_semantic = normalize_scores(semantic_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    
    hybrid_scores = (alpha * norm_semantic) + ((1 - alpha) * norm_bm25)
    return hybrid_scores


# =====================================================================
# 3. Hierarchical Search Engine
# =====================================================================
class HierarchicalRAGEngine:
    def __init__(self, tree_db, dense_model):
        self.tree_db = tree_db
        self.dense_model = dense_model
        self.categories = list(tree_db.keys())
        self.category_summaries = [tree_db[cat]["summary"] for cat in self.categories]

    def retrieve(self, query, alpha=0.5, top_k_leaves=1):
        """
        Executes a 2-stage hierarchical hybrid search.
        Stage 1: Identify best parent node.
        Stage 2: Search only inside the selected branch's leaf nodes.
        """
        print(f"\n[Stage 1] Querying parent categories...")
        cat_scores = compute_hybrid_scores(query, self.category_summaries, self.dense_model, alpha)
        best_cat_idx = np.argmax(cat_scores)
        selected_category = self.categories[best_cat_idx]
        print(f" -> Chosen Category: '{selected_category}' (Score: {cat_scores[best_cat_idx]:.4f})")
        
        print(f"[Stage 2] Searching within '{selected_category}' leaves...")
        leaves = self.tree_db[selected_category]["leaves"]
        leaf_texts = [leaf["text"] for leaf in leaves]
        
        leaf_scores = compute_hybrid_scores(query, leaf_texts, self.dense_model, alpha)
        ranked_indices = np.argsort(leaf_scores)[::-1][:top_k_leaves]
        
        retrieved_results = []
        for idx in ranked_indices:
            retrieved_results.append({
                "id": leaves[idx]["id"],
                "text": leaf_texts[idx],
                "score": leaf_scores[idx]
            })
            print(f" -> Found Chunk [{leaves[idx]['id']}]: \"{leaf_texts[idx][:70]}...\" (Score: {leaf_scores[idx]:.4f})")
            
        return retrieved_results, selected_category


# =====================================================================
# 4. LLM Generation
# =====================================================================
class LocalGenerator:
    """Manages an ultra-lightweight, CPU-friendly instruction LLM."""
    def __init__(self):
        print("\n[Init] Downloading and loading local LLM (HuggingFaceTB/SmolLM-135M-Instruct)...")
        # SmolLM-135M is roughly 270MB. It runs well on typical laptop CPUs.
        self.generator = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM-135M-Instruct",
            device="cpu"
        )
        print("[Init] LLM loaded successfully.")

    def generate(self, query, context):
        prompt_content = (
            "You are a helpful assistant. Use only the provided context to answer the question.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        messages = [{"role": "user", "content": prompt_content}]
        prompt = self.generator.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.generator(
            prompt, 
            max_new_tokens=80, 
            do_sample=False,  # Deterministic output
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        response = outputs[0]["generated_text"]
        return response[len(prompt):].strip()


# =====================================================================
# 5. Runtime Execution
# =====================================================================
if __name__ == "__main__":
    # A tiny hierarchical database representing Categories -> Specific Facts
    knowledge_tree = {
        "Technology": {
            "summary": "Topics covering software development, computer hardware, quantum systems, and artificial intelligence.",
            "leaves": [
                {"id": "tech_1", "text": "Artificial Intelligence is shifting towards agentic workflows using semantic reasoning and RAG."},
                {"id": "tech_2", "text": "Modern system-on-chip architectures package GPU, CPU, and NPU cores together to save battery."},
                {"id": "tech_3", "text": "Quantum computers use superconductive qubits operating near absolute zero to calculate complex equations."}
            ]
        },
        "Health": {
            "summary": "Topics discussing human biology, chronic illnesses, nutritional science, and physical exercise.",
            "leaves": [
                {"id": "health_1", "text": "High intake of refined sugars triggers insulin resistance, chronic cell inflammation, and diabetes."},
                {"id": "health_2", "text": "Regular zone 2 cardiovascular exercise increases mitochondrial density and overall heart resilience."},
                {"id": "health_3", "text": "Leafy green vegetables provide vital micronutrients like folate, iron, and vitamin K."}
            ]
        }
    }

    # Load small 80MB embedding model
    print("[Init] Loading local embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Initialize RAG engine and local generator
    engine = HierarchicalRAGEngine(knowledge_tree, embed_model)
    generator = LocalGenerator()

    # Define a query
    user_query = "What happens if I eat too much sugar?"
    print(f"\n======================================")
    print(f"User Query: '{user_query}'")
    print(f"======================================")

    # Retrieve matching context hierarchically
    retrieved_items, category = engine.retrieve(user_query, alpha=0.5, top_k_leaves=1)
    
    # Construct Context
    context_text = "\n".join([item["text"] for item in retrieved_items])

    # Generate answer
    print("\n[RAG] Generating response with LLM...")
    answer = generator.generate(user_query, context_text)
    
    print("\n================ ANSWER ================")
    print(answer)
    print("========================================")