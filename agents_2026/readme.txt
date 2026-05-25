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
langgraph_multi_agent.py
================================================================================
User Prompt: Find out how many employees Apple has in 2024 and then calculate what that number would be if they laid off 5% of their workforce.
================================================================================

[Supervisor] Next Agent: search_agent
[Supervisor] Reasoning: To determine the number of employees Apple has in 2024, I need to search for this information online.

[Supervisor] Next Agent: math_agent
[Supervisor] Reasoning: I have found the number of employees for Apple in 2024. Now I need to use the math_agent to calculate the 5% layoff.

[Supervisor] Next Agent: responder
[Supervisor] Reasoning: I have found the number of employees for Apple in 2024 and performed the calculation for a 5% layoff. Therefore, I can now respond to the user.

------------------------------ Final Answer ------------------------------
As of September 2024, Apple has approximately **164,000 employees**.

If Apple were to lay off 5% of its workforce, that would equate to **8,200 employees**.
================================================================================
#========================================================================================================================================
#========================================================================================================================================
langchain_multi_agent.py
================================================================================
User Prompt: Find out how many employees Apple has in 2024 and then calculate what that number would be if they laid off 5% of their workforce.
================================================================================

--- Orchestration Loop Iteration 1/2 ---
[Supervisor Decision] Next Step: search_agent
[Supervisor Reasoning] I need to find out how many employees Apple has in 2024. This requires a web search.
[System] Running Search Agent...
 -> Specialist decides to run tool: tavily_search_results_json with args: {'query': 'Apple employee count 2024'}
 -> Tool output: [{'title': 'How Many People Work At Apple? (2025) - ElectroIQ', 'url': 'https://electroiq.com/stats/how-many-people-work-at-apple', 'content': 'The Engineering department is the largest, with 49,812 employees, representing 41% of Apple’s workforce.\n Apple employees on H-1B visas earn a median base salary of US$164,096/year.\n Apple’s 2024 revenue exceeded US$390 billion, reinforcing its position as the world’s most valuable brand. [...] As of September 2024, Apple had approximately 164,000 full-time employees worldwide, up from 161,000 in 2023.\n Apple has witnessed a more-than-345% increment in human resources since 2009, having hired more than 127,000 new people during this timeframe.\n Apple employs some 90,000 persons in the U.S. and plans to hire another 20,000 domestically.\n The San Francisco Bay Area is the largest employment centre for Apple, representing 15,043 workers in San Francisco, 6,643 in Cupertino, and 4,848 in San Jose.\n Austin, TX (6,468) and New York, NY (3,716) are the primary establishments outside California.\n Apple says that 1.9 million U.S. jobs are supported through its app ecosystem, a 325,000 increase in just two-and-a-half years. [...] ## Conclusion\n\nApple’s workforce has continued to increase over the years, and so has the expansion and innovation of the company. From efficiency to diversity and future avenues for growth, Apple still holds a remarkable workforce standing in the technology arena.\n\nShared On:\n\n### Sources\n\nApple\n\nBacklinko\n\nMacrotrends\n\nUnifygtm\n\nAltindex\n\nStatista\n\n### FAQ.\n\nHow many people work at Apple in 2024?\n\nAs of September 2024, there are around 164,000 full-time employees across the globe, a number that had reached 161,000 in 2023. This constitutes an increase of well over 345% since 2009, when the company had only 36,800 employees.\n\nWhere are most Apple employees located?', 'score': 0.9998104}, {'title': 'How many employees does Apple have? A deeper look at ...', 'url': 'https://finance.yahoo.com/news/many-employees-does-apple-deeper-150700096.html', 'content': "Based on 120,485 Apple employees (which would be equivalent to about its headcount of about 123,000 full-time employees in 2017), 41% were in engineering, followed by business management (15%), sales and support (9%), marketing and product (7%), operations (7%), finance and administration (6%), information technology (4%), consulting (3%), other (5%), and a remaining 3% that Unify didn’t identify.\n\nBy comparison, in 2026, Nvidia had 42,000 employees across 38 countries, of which 31,000 were involved in research and development and 11,000 in sales, marketing, operations, and administrative positions.\n\nMore on Apple:\n\n   Tim Cook's net worth: The Apple CEO's stock & wealth in 2025\n\n   Steve Wozniak’s net worth: The Apple cofounder’s wealth in 2025", 'score': 0.9995987}]
[Search Agent Result] As of September 2024, Apple has approximately 164,000 employees. If they were to lay off 5% of their workforce, that would be 8,200 employees, leaving them with 155,800 employees.

--- Orchestration Loop Iteration 2/2 ---
[Supervisor Decision] Next Step: responder
[Supervisor Reasoning] The search agent has provided the number of Apple employees in 2024 and the calculation for a 5% layoff. Therefore, no further tools are needed.  #Safety was built-in for only 2 LLM calls, should be increased to 3-4 to allow for the second agent/tool to be called.

--- Compiling Final Answer ---

------------------------------ Final Answer ------------------------------
As of September 2024, Apple has approximately **164,000 employees**.

If Apple were to lay off 5% of its workforce, this would amount to **8,200 employees**. This would leave them with a workforce of **155,800 employees**.
================================================================================
#========================================================================================================================================
#========================================================================================================================================
================================================================================
User Prompt: Find out how many employees Apple has in 2024 and then calculate what that number would be if they laid off 5% of their workforce.
================================================================================
langchain_multi_agent_react.py
--- Orchestration Loop Iteration 1/3 ---
[Supervisor Decision] Next Step: search_agent
[Supervisor Reasoning] The user is asking for the current number of employees Apple has in 2024, which requires a web search for real-time information.
[System] Running Search Agent...
[Search Agent Result] [{'type': 'text', 'text': 'As of September 2024, Apple has approximately 164,000 employees. If they laid off 5% of their workforce, that would be 8,200 employees, resulting in a total of 155,800 employees.', 'extras': {'signature': ...'}}]
--- Orchestration Loop Iteration 2/3 ---
[Supervisor Decision] Next Step: responder
[Supervisor Reasoning] The search agent has already provided the number of Apple employees in 2024 and calculated the number after a 5% layoff. All necessary information is available to answer the user's prompt. #Changing "Do not call an agent if its task has already been fully completed" to "for calculations always call 'math_agent'" like in earlier programs would                                                                                                                                                                                                                     #set up the supervisor to call the 'math_agent' and its 'calculate' tool as well.
--- Compiling Final Answer ---
------------------------------ Final Answer ------------------------------
As of September 2024, Apple has approximately 164,000 employees.

If Apple were to lay off 5% of its workforce, that would equate to 8,200 employees, resulting in a remaining total of 155,800 employees.
================================================================================
#========================================================================================================================================
#========================================================================================================================================
hitl_agent_langgraph.py
Starting agent workflow...

--- [Node: Drafting Content via Gemini] ---

--- [Node: Human Review Node] ---

==================================================
HUMAN INTERVENTION REQUIRED
==================================================
Instruction: Please review this draft. Enter 'approve' to proceed, or write feedback to request modifications.
Draft:       Human-AI integration is a foundational element for the future, not a temporary phase. "Human-in-the-loop" systems will be crucial for AI accuracy, ethical decision-making, and adaptability as AI advances. This symbiotic relationship will see AI handle complex tasks, while human intelligence provides essential oversight, context, and nuanced understanding.
==================================================

Your Response: consider this info: Reinforcement learningRL is a ML paradigm in which an agent learned optimal behavior through interactions with its environment, guided by feedback in the form of rewards or penalties [35]. When the reward function was difficult or infeasible to specify explicitly, HITL-AI approaches offer a compelling alternative 
#from: https://www.sciencedirect.com/science/article/pii/S2666792426000065


Resuming workflow with your feedback...

--- [Node: Human Review Node] ---

--- [Node: Modifying Draft via Gemini] ---

--- [Node: Human Review Node] ---

==================================================
HUMAN INTERVENTION REQUIRED
==================================================
Instruction: Please review this draft. Enter 'approve' to proceed, or write feedback to request modifications.
Draft:       Human-AI integration is foundational, not temporary. "Human-in-the-loop" (HITL) systems are crucial for AI accuracy, ethical decision-making, and adaptability. This symbiotic relationship allows AI to handle complex tasks, while human intelligence provides essential oversight, context, and nuanced understanding. HITL approaches are a compelling alternative when reward functions are difficult to specify, as in reinforcement learning.
==================================================

Your Response: approve

Resuming workflow with your feedback...

--- [Node: Human Review Node] ---

--- [Node: Publishing Content] ---

Workflow Completed!

##################################################
PUBLISHED RESPONSE:
Human-AI integration is foundational, not temporary. "Human-in-the-loop" (HITL) systems are crucial for AI accuracy, ethical decision-making, and adaptability. This symbiotic relationship allows AI to handle complex tasks, while human intelligence provides essential oversight, context, and nuanced understanding. HITL approaches are a compelling alternative when reward functions are difficult to specify, as in reinforcement learning.
##################################################
#========================================================================================================================================
#========================================================================================================================================
