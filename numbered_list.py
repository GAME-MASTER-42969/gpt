from langchain_openai import ChatOpenAI  # Use ChatOpenAI for chat models
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
import os
import json
import re  # Import regex module for sanitizing filenames
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the ListItem model
class ListItem(BaseModel):
    id: str        # The numbering (e.g., "1", "1.1", "1.1.1")
    title: str     # The title text (e.g., "Introduction", "Definition of a Computer")

# Function to parse the output into ListItem objects
def parse_list(output: str):
    list_items = []
    for line in output.splitlines():
        # Remove leading/trailing whitespace and skip empty lines.
        stripped = line.strip()
        if not stripped:
            continue

        # We assume the numbering is separated by a dot and space.
        # Example: "1. Introduction" or "1.1 Definition of a Computer"
        if ". " in stripped:
            id_part, title = stripped.split(". ", 1)
            # If there are further dots in the id, they will be preserved.
            list_items.append(ListItem(id=id_part.strip(), title=title.strip()))
        else:
            # If the expected format is not found, add the entire line as title.
            list_items.append(ListItem(id="", title=stripped))
    return list_items

# Function to generate and parse the list using the LLM
def generate_and_parse_list(user_prompt, topics, subtopics, model_name="gpt-4"):
    llm = ChatOpenAI(
        temperature=0.5,
        max_tokens=1500,
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name
    )
    prompt = PromptTemplate(
        input_variables=["topics", "subtopics", "user_prompt"],
        template="""  
        You are a research assistant whose primary task is to break down {user_prompt} into a structured, numbered list of {topics} and {subtopics}. Your output should follow a strategic, step-by-step guide format and be organized using a multi-level numbering system (e.g., 1, 1.1, 1.1.1, 1.2, 1.3, ... up to 10.11.12 if needed and beyond).

Guidelines:
1. Analyze the user prompt carefully to extract key themes, tasks, or steps.
2. Organize your response as a numbered list where:
   - Main topics are numbered as 1, 2, 3, etc.
   - Subtopics follow the format 1.1, 1.2, etc.
   - Deeper levels follow further decimal notation (e.g., 1.1.1, 1.1.2, etc.).
3. Your list should serve as a strategic guide, timeline, or set of steps, rather than an analytical or deeply interpretive essay.
4. Avoid over-analysis; keep the structure straightforward and directly related to the user’s prompt.
5. Ensure clarity and logical progression in the numbered format, so each point builds on the previous ones.

When a user provides a prompt, return only the structured numbered list according to these instructions."""
    )
    
    # Format the prompt with the input variables
    formatted_prompt = prompt.format(
        user_prompt=user_prompt,
        topics=topics,
        subtopics=subtopics
    )
    
    # Generate response using the LLM
    response = llm.invoke(formatted_prompt)
    
    # Parse the response content into ListItem objects
    response_content = response.content  # Access the content attribute
    items = parse_list(response_content)
    
    return items

# Function to sanitize filenames
def sanitize_filename(name):
    # Remove invalid characters for filenames on Windows
    return re.sub(r'[<>:"/\\|?*]', '', name)

# Main function
if __name__ == "__main__":
    # Prompt the user for the input JSON file
    input_filename = input("Enter the path to the input JSON file: ").strip()
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' does not exist.")
        exit(1)
    
    # Read the JSON file
    with open(input_filename, "r") as file:
        input_data = json.load(file)
    
    # Extract the user prompt from the JSON file
    project_name = input_data.get("project_name", "Untitled Project")
    plan = input_data.get("plan", "")
    user_prompt = f"{project_name} {plan}"
    
    # Define topics and subtopics
    topics = "topics"
    subtopics = "subtopics"
    model_name = "gpt-4"  # Specify the model name
    
    # Generate and parse the list
    items = generate_and_parse_list(user_prompt, topics, subtopics, model_name)
    
    # Generate the output filename
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f"{base_name}_numbered_list.json"
    
    # Prepare the output JSON structure
    output_data = {
        "project_name": project_name,
        "plan": plan,
        "numbered_list": [item.model_dump() for item in items]  # Use model_dump instead of dict
    }

    # Save the items to the JSON file
    with open(output_filename, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Items saved to {output_filename}")