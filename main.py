import google.generativeai as genai
import json

# Multi-shot prompting: provide multiple examples in the prompt

def build_prompt(api_response, criteria, examples):
    return f"""
You are an API evaluator. Given the following API response and evaluation criteria, assess the API’s performance.

API Response:
{api_response}

Evaluation Criteria:
{criteria}

Here are several examples of correct evaluations:
{examples}

Now, provide your evaluation as a JSON object with the following structure:
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

# Multi-shot examples
examples = """
Example 1:
{
  "correctness": "The API returns the correct user data as requested.",
  "efficiency": "The response is quick and contains only necessary information.",
  "scalability": "The structure supports adding more users and handling larger datasets."
}

Example 2:
{
  "correctness": "The API provides accurate results for the given query.",
  "efficiency": "Minimal latency and optimized data transfer.",
  "scalability": "Can efficiently process thousands of records without performance loss."
}
"""

prompt = build_prompt(api_response, criteria, examples)

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
# Multi-shot prompting means providing the AI with multiple examples to guide its response.
# In this code, the prompt includes two sample evaluations in JSON format.
# This helps the model learn the expected structure, style, and variability for its own output.
# Multi-shot prompting is useful when you want the model to generalize from several examples and produce more robust, context-aware responses.