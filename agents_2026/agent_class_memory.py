import os
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. STATEFUL MEMORY CLASS
# ==========================================
class SimplePythonMemory:
    """Manages the conversation history locally in VSCode using a basic list."""
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage):
        self.messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def clear(self):
        self.messages = []


# ==========================================
# 2. DEFINE THE MULTIPLICATION TOOL
# ==========================================
@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers together. Always use this tool for multiplication."""
    return a * b


# ==========================================
# 3. AGENT DEFINITION WITH SYSTEM PROMPT
# ==========================================
class CustomAgent:
    def __init__(self, api_key: str):
        # Use gemini-1.5-flash for the free tier
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0
        )
        
        # Bind the multiplication tool to our model
        self.tools = [multiply_numbers]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize our custom memory class
        self.memory = SimplePythonMemory()

        # System Prompt defining how the agent handles routing
        self.system_message = SystemMessage(
            content=(
                "You are a helpful assistant with access to a multiplication tool.\n\n"
                "Guidelines:\n"
                "1. For general or conversational questions (including geography or general chat), "
                "answer them yourself directly.\n"
                "2. For any math, calculation, or multiplication questions, you MUST call the "
                "`multiply_numbers` tool. Do not try to compute the multiplication yourself."
            )
        )

    def run(self, user_input: str) -> str:
        # Step A: Save the user's message to the custom local memory
        self.memory.add_message(HumanMessage(content=user_input))

        # Step B: Prepend the system prompt instruction to the memory history
        full_payload = [self.system_message] + self.memory.get_messages()

        # Step C: Ask the LLM
        response = self.llm_with_tools.invoke(full_payload)
        
        # Save the LLM's response to memory
        self.memory.add_message(response)

        # Step D: Execute tool calls if the LLM requests them
        if response.tool_calls:
            print("\n[System: Agent decided to call the multiplication tool...]")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "multiply_numbers":
                    # Run the tool locally
                    result = multiply_numbers.invoke(tool_args)
                    print(f"[System: Tool run -> {tool_name}({tool_args}) = {result}]")

                    # Record the tool's result in the memory
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                    self.memory.add_message(tool_message)

            # Step E: Send the updated history (with the tool result) back to the LLM
            updated_payload = [self.system_message] + self.memory.get_messages()
            final_response = self.llm_with_tools.invoke(updated_payload)
            
            # Save final conversational response to memory
            self.memory.add_message(final_response)
            return final_response.content
        
        else:
            return response.content


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure you set your API key
    api_key = os.environ.get("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY_HERE"
    
    if api_key == "YOUR_GOOGLE_API_KEY_HERE":
        print("Please replace 'YOUR_GOOGLE_API_KEY_HERE' with your actual free tier Google API key.")
        exit(1)

    agent = CustomAgent(api_key=api_key)

    print("Agent initialized. You can ask it to multiply numbers, or just chat with it.")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.strip().lower() == "exit":
            break
        
        response = agent.run(user_query)
        print(f"Agent: {response}\n")