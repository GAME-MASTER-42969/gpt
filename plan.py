from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env
load_dotenv()

def generate_response_and_project_name(user_prompt, model_name="gpt-4o", max_tokens=800):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a visionary CEO and strategic planner. Generate high-level, actionable project plans with clear objectives, major milestones, and concise executive summaries. Focus on strategy, impact, and resource allocation. Your output should reflect the perspective and priorities of a top executive."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.6,
    )
    # Extract project name and plan from response
    plan_text = response.choices[0].message.content.strip()
    project_name = plan_text.split('\n')[0]  # Example: first line as project name
    return project_name, plan_text

# Function to save the response to a JSON file
def save_to_json(project_name, response_text):
    # Remove invalid characters for file names
    sanitized_project_name = "".join(c for c in project_name if c.isalnum() or c in (" ", "_")).strip()
    filename = f"{sanitized_project_name.replace(' ', '_').lower()}_plan.json"
    data = {"project_name": project_name, "plan": response_text}
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Response saved to {filename}")

if __name__ == "__main__":
    # Example user prompt
    user_prompt = "I want to build a VTOL RC drone with omni wheels. Write a paragraph to explain the fundamentals of drones, then a paragraph for the VTOL drone, and finally create a plan for my custom RC VTOL drone with omnidirectional wheels."

    # Generate the response and project name
    project_name, response_text = generate_response_and_project_name(user_prompt, model_name="gpt-4o", max_tokens=2200)

    # Print the generated project name and response
    print(f"Generated Project Name: {project_name}")
    print("\nGenerated Response:")
    print(response_text)

    # Save the response to a JSON file
    save_to_json(project_name, response_text)