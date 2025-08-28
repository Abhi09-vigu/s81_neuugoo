import google.generativeai as genai

# Zero-shot prompt for API evaluation
prompt = """
You are an API evaluator. Given the following API response and evaluation criteria, assess the API’s performance.

API Response:
{
  "status": "success",
  "data": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ]
}

Evaluation Criteria:
- Correctness: Does it return the correct data based on the provided request?
- Efficiency: How quickly and efficiently does it fetch the data?
- Scalability: Can the API handle increased traffic and large data sets without performance degradation?

Provide a brief evaluation for each criterion.
"""

# Example usage with Gemini API (replace 'your-api-key' with your actual key)
genai.configure(api_key="AIzaSyCu1GmUKXZEuIzgUToYTXleaP-WTqLhAvA")
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content(prompt)

print(response.text)

# --- Video Explanation ---
# Zero-shot prompting means giving an AI a task using only instructions, without any examples.
# In this code, the prompt asks the AI to evaluate an API response using specific criteria.
# No sample answers are provided, so the AI must understand and perform the task from the instructions alone.
# This demonstrates zero-shot prompting in action, where the AI applies its learned knowledge to a new task.