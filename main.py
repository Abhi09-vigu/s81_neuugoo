import google.generativeai as genai
import json

# Chain of thought prompting: ask the model to reason step by step

def build_prompt(api_response, criteria):
    return f"""
You are an API evaluator. Given the following API response and evaluation criteria, assess the API’s performance.

API Response:
{api_response}

Evaluation Criteria:
{criteria}

Think step by step. First, analyze the API response for correctness. Next, evaluate its efficiency. Then, consider scalability. After reasoning through each criterion, summarize your evaluation as a JSON object with the following structure:
{{
  "correctness": "<your assessment>",
  "efficiency": "<your assessment>",
  "scalability": "<your assessment>"
}}
<END>
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
    }
)

print("Raw AI response:")
print(response.text)

# Try to parse structured output as JSON
try:
    cleaned_text = response.text.split("<END>")[0].strip()
    structured_output = json.loads(cleaned_text)
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
# Chain of thought prompting means instructing the AI to reason step by step before giving a final answer.
# In this code, the prompt asks the model to analyze correctness, efficiency, and scalability one by one, then summarize its findings in a structured JSON output.
# This approach helps the model produce more accurate, logical, and transparent responses by making its reasoning process