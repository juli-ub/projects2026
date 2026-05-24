**All build with the help of Google AI Studio**
#========================================================================================================================================
#========================================================================================================================================
python tavily_calculator.py
--- Running Multi-Turn Conversation (Max 3 turns) ---
The agent now has access to search and calculation tools.

[Turn 1/3] Ask a question: What is the capital of France?

User: What is the capital of France?
============================================================

[Step: model]
Paris. I didn't need to use any tools.

============================================================


[Turn 2/3] Ask a question: What is the temperature there?

User: What is the temperature there?
============================================================

[Step: model]
⚙️ Action: Calling 'tavily_search' with args: {'query': 'current temperature in Paris'}

[Step: tools]
{"query": "current temperature in Paris", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.weatherbug.com/weather-forecast/now/paris-ile-del-france-fr", "title": "Paris, Ile-del-France, FR Weather Today & Tomorrow - WeatherBug", "content": "Source Weather Station:PARIS/LE BOURGET. Current Temperature: 57°. Feels like:57°. Today's high temperature: Hi: 63°. Today's low temperature: Lo: 57°. Current", "score": 0.8512743, "raw_content": null}, {"url": "https://www.theweathernetwork.com/en/city/fr/ile-de-france/paris/current", "title": "Paris, IDF, FR Current Weather - The Weather Network", "content": "What is the current temperature in Paris? 13°. What is the “feels like” temperature in Paris? 13°. What is the high and low temperature today in Paris? 26°. 15", "score": 0.79958355, "raw_content": null}, {"url": "https://www.timeanddate.com/weather/france/paris/ext", "title": "Paris 14 Day Extended Forecast - Weather - Time and Date", "content": "Night Sky · TodayHourly14 DaysPastClimate. Currently: 57 °F. Sunny. (Weather station: Villacoublay, France). See more current weather. ×. Advertising: Content", "score": 0.78944516, "raw_content": null}], "response_time": 1.02, "request_id": "8d9cd0c0-8813-4f4e-970e-252d01d8f255"}

[Step: model]
The current temperature in Paris is 57°F. I used the `tavily_search` tool to get this information.

============================================================


[Turn 3/3] Ask a question: If you would add 10 degrees to that temperature what would be the resulting temperature?

User: If you would add 10 degrees to that temperature what would be the resulting temperature?
============================================================

[Step: model]
⚙️ Action: Calling 'calculator' with args: {'a': 57, 'operation': 'add', 'b': 10}

[Step: tools]
67.0

[Step: model]
If you add 10 degrees to 57°F, the resulting temperature would be 67°F. I used the `calculator` tool to determine this.

============================================================


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


#========================================================================================================================================
#========================================================================================================================================
tavily_react_agent.py
Sending Query to Agent: 'Who is the current Prime Minister of the United Kingdom?'

Agent is reasoning and executing tools...

========================================
AGENT RESPONSE:
========================================
My Initial Hypothesis: I believe Rishi Sunak is the current Prime Minister of the United Kingdom.

Verification & Search Findings: I performed a web search for "current Prime Minister of the United Kingdom." The search results confirm that Rishi Sunak is indeed the current Prime Minister, having taken office in October 2022.

Correction & Final Verified Answer: The current Prime Minister of the United Kingdom is Rishi Sunak.

#========================================================================================================================================
#========================================================================================================================================
langgraph_conditional_edge.py
==================================================
RUNNING PROMPT: 'Calculate the age of the current President of France multiplied by 2.'
==================================================

--- START NODE ---

--- PLAN NODE ---
Plan: Plan: Search for the current President of France and their birthdate to determine their age.

--- ROUTER NODE ---
Decision: 'search' (Iteration: 1/3)

--- SEARCH NODE ---
Result: {'query': 'Calculate the age of the current President of France multiplied by 2.', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'url': 'https://en.wikipedia.org/wiki/Emmanuel_Macron', 'title': 'Emmanuel Macron - Wikipedia', 'content': 'Emmanuel Jean-Michel Frédéric Macron (born 21 December 1977) is a French politician who has served as President of France and Co-Prince of Andorra since 2017.', 'score': 0.572078, 'raw_content': None}], 'response_time': 1.08, 'request_id': 'e49f4c3f-b337-41a4-be83-22eb773bbcd5'}

--- PLAN NODE ---
Plan: Plan: Calculate Emmanuel Macron's current age based on his birthdate (21 December 1977) and then multiply it by 2.

--- ROUTER NODE ---
Decision: 'calc' (Iteration: 2/3)

--- CALCULATOR NODE ---
Result: * 2 = Error evaluating expression '* 2': invalid syntax (<string>, line 1)

--- PLAN NODE ---
Plan: Use the calculator to determine Emmanuel Macron's current age based on his birthdate (21 December 1977) and then multiply that age by 2.

--- ROUTER NODE ---
⚠️ Loop limit reached! Forcing state machine to finish.

--- FINISH NODE ---
#========================================================================================================================================
#========================================================================================================================================



