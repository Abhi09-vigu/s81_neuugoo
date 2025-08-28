import google.generativeai as genai
import json

# --- RTFC Framework ---
# RTFC stands for Role, Task, Format, and Constraints.
# It helps design clear, effective prompts for LLMs.

# System prompt (Role, Task, Format, Constraints)
system_prompt = """
You are Neuugoo, an expert AI tutor and evaluator for personalized learning platforms.
Your task is to assess API responses for correctness, efficiency, and scalability.
Format your output as a JSON object with fields: correctness, efficiency, scalability.
Constraints: Be concise, objective, and use only information from the provided API response and criteria.
"""

# User prompt (provides context and request)
def build_user_prompt(api_response, criteria):
    return f"""
API Response:
{api_response}

Evaluation Criteria:
{criteria}

Please provide your evaluation.
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

user_prompt = build_user_prompt(api_response, criteria)

# Example usage with Gemini API (replace 'your-api-key' with your actual key)
genai.configure(api_key="your-api-key")
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content(
    [system_prompt, user_prompt],
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
# The RTFC framework stands for Role, Task, Format, and Constraints.
# In this code, the system prompt sets the role (AI tutor/evaluator), task (assess API responses), format (JSON), and constraints (concise, objective, use only provided info).
# The user prompt provides the context and request.
# This structure ensures the LLM understands its job, how to respond, and any boundaries, resulting in clear and reliable