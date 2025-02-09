import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_response(query, context):
    """Generate a response using OpenAI's API."""
    print("\nContext passed:")
    print(context)
    print("\n")

    messages = [
        {"role": "system", "content": (
            "You are a Kubernetes and NGINX ingress expert. Validate the context for errors and deprecated features. "
            "Ensure responses adhere to the latest standards and provide necessary corrections."
        )},
        {"role": "user", "content": f"Given the following context:\n{context}\nAnswer this query:\n{query}\n"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=4090,
            temperature=0
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error: {e}")
        return "Error generating response."

if __name__ == "__main__":
    query = "How do I configure NGINX ingress?"
    response = generate_response(query, "NGINX config details...")
    print("GPT Response:", response)
