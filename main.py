import google.generativeai as genai
import json

def build_prompt(api_response, criteria):
    return f"""
You are an API evaluator. Given the following API response and evaluation criteria, assess the API’s performance.

API Response:
{api_response}

Evaluation Criteria:
{criteria}

Return your evaluation as a JSON object with the following structure:
{{
  "correctness": "<your assessment>",
  "efficiency": "<your assessment>",
  "scalability": "<your assessment>"
}}
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

# Build the prompt dynamically
prompt = build_prompt(api_response, criteria)

# Example usage with Gemini API (replace 'your-api-key' with your actual key)
genai.configure(api_key="your-api-key")
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content(
    prompt,
    generation_config={
        "top_p": 0.8,      # Top P parameter
        "temperature": 0.7, # Temperature parameter
        "top_k": 40        # Top K parameter added
    }
)

print("Raw AI response:")
print(response.text)

# Try to parse structured output as JSON
try:
    structured_output = json.loads(response.text)
    print("\nStructured Output:")
    print(json.dumps(structured_output, indent=2))
except Exception as e:
    print("\nCould not parse structured output as JSON.")
    print(f"Error: {e}")

# Log the number of tokens used (if available)
if hasattr(response, "usage_metadata") and "total_tokens" in response.usage_metadata:
    print(f"\nTokens used: {response.usage_metadata['total_tokens']}")
else:
    print("\nToken usage information not available.")

# --- Video Explanation ---
# Structured output in LLMs means instructing the model to return its response in a specific format, such as JSON.
# This makes it easier to parse, validate, and use the output programmatically.
# In this code, the prompt asks the AI to return its evaluation as a JSON object with defined fields.
# The code then attempts to parse the response as JSON, demonstrating structured output in action.