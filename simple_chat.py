import openai
import os

# Set the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def chat_with_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    while True:
        user_input = input("Enter your prompt (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break
        print(chat_with_llm(user_input))


