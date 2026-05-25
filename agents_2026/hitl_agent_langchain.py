import os
import getpass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. API Key Setup Check
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 2. Define standard LCEL Chains
# Drafting Chain
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional writer. Write a short introductory paragraph (maximum 3 sentences) about the requested topic."),
    ("human", "Topic: {topic}")
])
draft_chain = draft_prompt | llm | StrOutputParser()

# Redrafting Chain
redraft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert editor. Modify the current draft based strictly on the user's feedback. Return only the revised draft, keeping it brief and professional."),
    ("human", "Current Draft:\n{draft}\n\nFeedback:\n{feedback}")
])
redraft_chain = redraft_prompt | llm | StrOutputParser()

# 3. Custom Orchestration Loop
def run_orchestration_loop(topic: str):
    print("Starting LangChain-only custom orchestrated agent...")
    
    # Step 1: Generate initial draft
    print("\n--- [Drafting Content via Gemini] ---")
    draft = draft_chain.invoke({"topic": topic})
    
    # Step 2: Loop for Human Review
    while True:
        print("\n" + "="*50)
        print("HUMAN INTERVENTION REQUIRED")
        print("="*50)
        print(f"Draft: {draft}")
        print("="*50)
        
        user_input = input("\nEnter feedback (or type 'approve' to proceed): ")
        
        # Check approval condition
        if user_input.strip().lower() == 'approve':
            print("\nDraft approved!")
            break
            
        # Re-draft based on feedback and repeat
        print("\n--- [Modifying Draft via Gemini] ---")
        draft = redraft_chain.invoke({
            "draft": draft,
            "feedback": user_input
        })
        
    print("\nWorkflow Finished!")
    print("\n" + "#"*50)
    print(f"PUBLISHED RESPONSE:\n{draft}")
    print("#"*50)

if __name__ == "__main__":
    run_orchestration_loop("The evolution of Human-in-the-loop AI workflows")