


#========================================================================================================================================
#========================================================================================================================================
python self_correcting_code_generation.py

--- Starting Self-Correcting Agent ---

[Attempt 1/3] Requesting code from Gemini...
Running locally executed tests...
❌ Unit Test Failed.
Feedback/Error message sent to agent:
Test B Failed: For input '{"x": "string_to_ignore", "y": 5.5, "z": [10, "skip_me", true]}', expected sum to be 15.5, but got 16.5.


[Attempt 2/3] Requesting code from Gemini...
Running locally executed tests...

🎉 Success! The code successfully passed all unit tests.
--------------------------------------------------
import json

def parse_and_sum(json_string):
    """
    Parses a JSON string and calculates the sum of all numerical values recursively.

    Args:
        json_string: A string containing valid JSON.

    Returns:
        The sum of all numerical values found in the JSON structure, or 0 if
        the JSON is invalid or cannot be parsed.
    """
    try:
        data = json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return 0

    def recursive_sum(item):
        total = 0
        # Strictly check for int and float, excluding booleans
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            total += item
        elif isinstance(item, dict):
            for key, value in item.items():
                # Sum numeric keys as well, excluding booleans
                if isinstance(key, (int, float)) and not isinstance(key, bool):
                    total += key
                total += recursive_sum(value)
        elif isinstance(item, list):
            for element in item:
                total += recursive_sum(element)
        # Ignore other types like strings, booleans, None
        return total

    return recursive_sum(data)

#========================================================================================================================================
#========================================================================================================================================
tavily_self_correcting_agent.py:
--- Processing Query: 'Who won the men's singles title at the most recent Wimbledon championships?' ---

[1/4] Generating initial answer...
Initial LLM Answer:
The most recent Wimbledon men's singles title was won by **Carlos Alcaraz**.

[2/4] Fetching search results from Tavily...

[3/4] Evaluating initial answer against ground truth...
Is Accurate? False
Evaluation Reason: The LLM answer states that Carlos Alcaraz won the most recent Wimbledon men's singles title. However, the provided search results indicate that Jannik Sinner won the 2025 Wimbledon men's singles title, defeating Carlos Alcaraz.

[4/4] Inaccuracy detected. Generating corrected answer...

Final Corrected Answer:
The most recent Wimbledon men's singles title was won by **Jannik Sinner**. He defeated Carlos Alcaraz of Spain in four sets in the 2025 championships.

