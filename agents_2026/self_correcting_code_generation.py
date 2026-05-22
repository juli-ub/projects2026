import os
import re
import getpass
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. API Key & Model Configuration
# ==========================================
if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI Studio API key: ")

# Initialize the Gemini Model (gemini-1.5-flash is fast and suited for this task)
# Setting temperature low makes code generation more deterministic
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)


# ==========================================
# 2. Local Code Execution & Test Harness
# ==========================================
def extract_code(text: str) -> str:
    """Extracts raw python code from Markdown blocks if present."""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback for generic code block
    pattern_alt = r"```(.*?)```"
    match_alt = re.search(pattern_alt, text, re.DOTALL)
    if match_alt:
        return match_alt.group(1).strip()
        
    return text.strip()


def run_tests(code_str: str) -> tuple[bool, str]:
    """
    Executes the LLM-generated code locally and runs unit tests.
    Returns (success_boolean, feedback_string).
    """
    try:
        # Define a single namespace for both globals and locals
        execution_context = {}
        exec(code_str, execution_context)
        
        # 1. Verify function exists
        if 'parse_and_sum' not in execution_context:
            return False, "Error: You must define a function named 'parse_and_sum'."
        
        parse_and_sum = execution_context['parse_and_sum']
        
        # 2. Test Case A: Nested dictionary and list traversal
        test_input_a = '{"a": 1, "b": {"c": 2, "d": [3, 4]}}'
        res_a = parse_and_sum(test_input_a)
        if res_a != 10:
            return False, f"Test A Failed: For input '{test_input_a}', expected sum to be 10, but got {res_a}."
            
        # 3. Test Case B: Handing mixed values (must ignore non-numeric types)
        test_input_b = '{"x": "string_to_ignore", "y": 5.5, "z": [10, "skip_me", true]}' # true is interpreted as 1 which causes the error
        res_b = parse_and_sum(test_input_b)
        if res_b != 15.5:
            return False, f"Test B Failed: For input '{test_input_b}', expected sum to be 15.5, but got {res_b}."
            
        # 4. Test Case C: Malformed JSON string (must return 0 gracefully)
        test_input_c = '{"broken_json": '
        res_c = parse_and_sum(test_input_c)
        if res_c != 0:
            return False, f"Test C Failed: For invalid JSON '{test_input_c}', expected return value 0, but got {res_c}."
            
        return True, "All tests passed successfully!"
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return False, f"Runtime execution failed with the following traceback:\n{error_trace}"


# ==========================================
# 3. Self-Correcting Agent Loop
# ==========================================
def run_agent():
    # Prompt the system to act as a code writer
    system_instruction = (
        "You are an expert Python developer. Your goal is to write a single Python function "
        "according to the specifications. Only output Python code inside a markdown code block "
        "like this:\n"
        "```python\n"
        "# your code\n"
        "```\n"
        "Do not write explanations, markdown text, or descriptions outside of the code block."
    )

    # The programming assignment
    task_instruction = (
        "Write a Python function named 'parse_and_sum' that accepts a JSON string as an argument. "
        "The function must parse the JSON and calculate the sum of all numerical values (integers and floats) "
        "recursively found anywhere inside the structure (keys, list elements, nested dicts). "
        "Important specs:\n"
        "1. Ignore non-numeric data types (strings, booleans like True/False, null/None values).\n"
        "2. If the JSON string is invalid or cannot be parsed, return 0 instead of throwing an error."
    )

    # Initialize conversation history with LangChain messages
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=task_instruction)
    ]

    max_attempts = 3
    
    print("\n--- Starting Self-Correcting Agent ---")
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n[Attempt {attempt}/{max_attempts}] Requesting code from Gemini...")
        
        # Invoke Gemini via LangChain
        response = llm.invoke(messages)
        
        # Save assistant's raw response to conversation history
        messages.append(AIMessage(content=response.content))
        
        # Parse output
        raw_code = extract_code(response.content)
        
        print("Running locally executed tests...")
        success, feedback = run_tests(raw_code)
        
        if success:
            print("\n🎉 Success! The code successfully passed all unit tests.")
            print("-" * 50)
            print(raw_code)
            print("-" * 50)
            return
        else:
            print(f"❌ Unit Test Failed.")
            print(f"Feedback/Error message sent to agent:\n{feedback}\n")
            
            if attempt == max_attempts:
                print("⚠️ Maximum attempts reached. Agent failed to self-correct in time.")
                return
            
            # Formulate the self-correction prompt and add it to the history
            correction_prompt = (
                f"Your previous code failed our verification tests.\n\n"
                f"--- Error / Feedback ---\n"
                f"{feedback}\n"
                f"------------------------\n\n"
                f"Please review your code, fix the logic or syntax errors, and output the updated version."
                f"Advice 1:" #this advies fixes the error in test B because theboolean is interpreted as 1. This is intended so that the correction prompt leads to success. 
                f"1. Ignore non-numeric data types (strings, booleans like True/False, null/None values). "
                f"NOTE: In Python, booleans (True/False) are subclasses of integers. To prevent booleans "
                f"from being counted as numbers, check types strictly (e.g., 'type(val) in (int, float)' "
                f"or 'isinstance(val, (int, float)) and not isinstance(val, bool)').\n"
            )
            messages.append(HumanMessage(content=correction_prompt))


if __name__ == "__main__":
    run_agent()