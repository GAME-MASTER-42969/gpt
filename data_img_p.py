from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import os
import json
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Define the ListItem model
class ListItem(BaseModel):
    id: str
    title: str
    explanation: str = None
    img_prompt: str = None

# Function to generate explanations
def generate_explanation(title: str, model_name="gpt-4", max_tokens=120) -> str:
    llm = ChatOpenAI(
        temperature=0.7,
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=max_tokens
    )
    prompt = PromptTemplate(
        input_variables=["title"],
        template=(
            "Provide a detailed and specific explanation for the following topic or subtopic: '{title}'. "
            "The explanation should focus on the key aspects, details, and context of the topic, ensuring it is informative and relevant and should not exceed 10 lines/sentences."
        )
    )
    formatted_prompt = prompt.format(title=title)
    response = llm.invoke(formatted_prompt)
    return response.content.strip() if response.content else "No explanation generated."

# Function to generate image prompts
def generate_image_prompt(title: str, explanation: str, model_name="gpt-4", max_tokens=69) -> str:
    llm = ChatOpenAI(
        temperature=0.7,
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=max_tokens
    )
    prompt = PromptTemplate(
        input_variables=["title", "explanation"],
        template=(
            "Based on the title '{title}' and the explanation '{explanation}', generate a detailed and descriptive image prompt "
            "suitable for an AI image or 3d model generator. The prompt should describe the visual elements, context, and style of the image or infographic."
        )
    )
    formatted_prompt = prompt.format(title=title, explanation=explanation)
    response = llm.invoke(formatted_prompt)
    return response.content.strip() if response.content else "No image prompt generated."

# Function to process the JSON file
def process_json(input_file: str, output_file: str, explanation_tokens=120, img_prompt_tokens=69):
    # Read the input JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    if not data:
        print("Input file is empty. Exiting.")
        return

    print(f"Processing file: {input_file}")

    # Extract project_name, plan, and numbered_list
    project_name = data.get("project_name", "Untitled Project")
    plan = data.get("plan", "No plan provided.")
    numbered_list = data.get("numbered_list", [])

    if not numbered_list:
        print("No numbered list found in the input file. Exiting.")
        return

    print(f"Project Name: {project_name}")
    print(f"Plan: {plan[:100]}...")  # Print the first 100 characters of the plan for brevity
    print(f"Number of items in numbered list: {len(numbered_list)}")

    # Process each item in the numbered list
    processed_list = []
    for idx, item in enumerate(numbered_list, start=1):
        title = item.get("title", "Untitled")
        id = item.get("id", "")

        print(f"Processing item {idx}/{len(numbered_list)}: {id} - {title}")

        # Generate explanation
        explanation = generate_explanation(title, max_tokens=explanation_tokens)

        # Generate image prompt
        img_prompt = generate_image_prompt(title, explanation, max_tokens=img_prompt_tokens)

        # Create a new ListItem object
        list_item = ListItem(
            id=id,
            title=title,
            explanation=explanation,
            img_prompt=img_prompt
        )
        processed_list.append(list_item.model_dump())

    # Prepare the output JSON structure
    output_data = {
        "project_name": project_name,
        "plan": plan,
        "numbered_list": processed_list
    }

    # Save the processed data to the output JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Processed data saved to {output_file}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.completions.create(
        model="text-davinci-003",
        prompt="Your prompt",
        max_tokens=100
    )
    result = response.choices[0].text

if __name__ == "__main__":
    # Input and output file paths
    input_file = input("Enter the path to the input JSON file: ").strip()
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        exit(1)

    output_file = f"processed_{os.path.basename(input_file)}"

    # Process the JSON file with token limits for explanation and image prompt
    process_json(input_file, output_file, explanation_tokens=100, img_prompt_tokens=150)