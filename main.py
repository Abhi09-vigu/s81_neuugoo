import google.generativeai as genai
import json

# Define a function schema for function calling
function_schema = {
    "name": "evaluate_api",
    "description": "Evaluates an API response for correctness, efficiency, and scalability.",
    "parameters": {
        "type": "object",
        "properties": {
            "correctness": {"type": "string", "description": "Assessment of correctness."},
            "efficiency": {"type": "string", "description": "Assessment of efficiency."},
            "scalability": {"type": "string", "description": "Assessment of scalability."}
        },
        "required": ["correctness", "efficiency", "scalability"]
    }
}

def build_prompt(api_response, criteria):
    return f"""
You are an API evaluator. Given the following API response and evaluation criteria, assess the API’s performance.

API Response:
{api_response}

Evaluation Criteria:
{criteria}
"""

# Example dynamic input
api_response = """
{
  "status": "success",
  "data": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ]
}
"""

criteria = """
- Correctness: Does it return the correct data based on the provided request?
- Efficiency: How quickly and efficiently does it fetch the data?
- Scalability: Can the API handle increased traffic and large data sets without performance degradation?
"""

prompt = build_prompt(api_response, criteria)

# Example usage with Gemini API (replace 'your-api-key' with your actual key)
genai.configure(api_key="your-api-key")
model = genai.GenerativeModel("gemini-pro")

response = model.generate_content(
    prompt,
    generation_config={
        "top_p": 0.8,
        "temperature": 0.7,
        "top_k": 40,
        "stop_sequences": ["<END>"]
    },
    tools=[function_schema],  # Function calling schema
    tool_config={"function_call": "evaluate_api"}  # Instruct model to call the function
)

print("Raw AI response:")
print(response.text)

# Try to extract function call arguments if present
if hasattr(response, "function_call") and response.function_call:
    print("\nFunction Call Arguments:")
    print(json.dumps(response.function_call["arguments"], indent=2))
else:
    print("\nNo function call arguments found.")

# Log the number of tokens used (if available)
if hasattr(response, "usage_metadata") and "total_tokens" in response.usage_metadata:
    print(f"\nTokens used: {response.usage_metadata['total_tokens']}")
else:
    print("\nToken usage information not available.")

# --- Video Explanation ---
# Function calling in LLMs allows the model to return structured outputs that can trigger specific functions in code.
# The model is given a schema describing the function and its parameters.
# When the model's output matches the schema, the function can be called programmatically with the returned arguments.
# This enables seamless integration between LLMs and software systems, automating tasks and workflows.
# In this code, the Gemini model is instructed to call the 'evaluate_api' function and return its arguments in a structured way.