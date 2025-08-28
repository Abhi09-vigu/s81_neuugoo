import google.generativeai as genai

def build_prompt(api_response, criteria):
    return f"""
You are an API evaluator. Given the following API response and evaluation criteria, assess the API’s performance.

API Response:
{api_response}

Evaluation Criteria:
{criteria}

Provide a brief evaluation for each criterion.
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
response = model.generate_content(prompt)

print(response.text)

# Log the number of tokens used (if available)
if hasattr(response, "usage_metadata") and "total_tokens" in response.usage_metadata:
    print(f"Tokens used: {response.usage_metadata['total_tokens']}")
else:
    print("Token usage information not available.")

# --- Video Explanation ---
# Tokens are chunks of text (words or parts of words) that AI models use to process and generate language.
# Each prompt and response is broken down into tokens, and the number of tokens affects cost and performance.
# In this code, after every AI call, the number of tokens used is logged to the console for transparency.