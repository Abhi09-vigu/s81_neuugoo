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
        "top_k": 40,        # Top K parameter
        "stop_sequences": ["<END>"] # Stop sequence added
    }
)

print("Raw AI response:")
print(response.text)

# Try to parse structured output as JSON
try:
    # Remove the stop sequence if present
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

# --- Embeddings Example ---
# Generate embeddings for a sample text using Gemini API
sample_text = "Neuugoo is an AI-powered interactive learning ecosystem."

embedding_model = genai.GenerativeModel("embedding-001") # Use Gemini's embedding model
embedding_response = embedding_model.embed_content([sample_text])

print("\nEmbedding vector for sample text:")
print(embedding_response["embedding"])

# --- Video Explanation ---
# Embeddings in LLMs are high-dimensional numerical representations of text.
# They capture the semantic meaning of words, sentences, or documents, allowing models to compare and analyze text efficiently.
# Embeddings are computed by passing text through a neural network, resulting in a vector.
# Practical applications include semantic search, clustering, recommendation systems, and measuring text similarity.
# In this code, we generate an embedding for a sample sentence, which can be used for these tasks.