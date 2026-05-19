import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types # Import types to construct the function response
from dotenv import load_dotenv

load_dotenv()
client_ai = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

async def run():
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Define the tool schema
            tools_schema = [{
                "function_declarations":[{
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"}, "b": {"type": "integer"}
                        },
                        "required": ["a", "b"]
                    }
                }]
            }]

            prompt = 'What is 15 plus 27?'
            print(f"User: {prompt}")
            
            # Turn 1: Ask Gemini
            response = client_ai.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config={"tools": tools_schema}
            )

            if response.function_calls:
                call = response.function_calls[0]
                print(f"Gemini wants to call: {call.name} with {call.args}")
                
                # Execute tool
                tool_result = await session.call_tool(call.name, arguments=dict(call.args))
                result_text = tool_result.content[0].text
                print(f"Tool Result: {result_text}")
                
                # Turn 2: Send the result back to Gemini
                # We provide the original prompt, the model's call, and our function response
                final_response = client_ai.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=[
                        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
                        types.Content(role="model", parts=[types.Part.from_function_call(name=call.name, args=call.args)]),
                        types.Content(role="user", parts=[types.Part.from_function_response(name=call.name, response={"result": result_text})])
                    ]
                )
                print(f"Gemini: {final_response.text}")
            else:
                print(f"Gemini: {response.text}")

if __name__ == "__main__":
    asyncio.run(run())