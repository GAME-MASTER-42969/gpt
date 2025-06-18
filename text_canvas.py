from plan import generate_response_and_project_name, save_to_json
from numbered_list import generate_and_parse_list, ListItem  # Import ListItem
from data_img_p import generate_explanation, generate_image_prompt
import json
import os
import re  # For sanitizing file names


def main():
    print("Welcome to the Researcher Assistant!")
    print("This script will generate a plan, a numbered list, and detailed explanations with image prompts.")
    
    # Step 1: Generate the project name and plan
    user_prompt = input("Enter your prompt for the plan: ").strip()
    if not user_prompt:
        print("Error: A user prompt is required.")
        return

    # Generate the project name and plan
    project_name, plan_text = generate_response_and_project_name(user_prompt, model_name="gpt-4", max_tokens=1500)

    # Sanitize the project name to create a valid file name
    sanitized_project_name = re.sub(r'[<>:"/\\|?*]', '', project_name).replace(" ", "_").replace("&", "and").lower()
    output_file = f"{sanitized_project_name}_processed.json"

    # Step 2: Generate the numbered list with explanations and image prompts
    print("Generating the numbered list with explanations and image prompts...")
    topics = "topics"
    subtopics = "subtopics"
    numbered_list_items = generate_and_parse_list(plan_text, topics, subtopics, model_name="gpt-4")

    processed_list = []
    for idx, item in enumerate(numbered_list_items, start=1):
        title = item.title
        id = item.id

        print(f"Processing item {idx}/{len(numbered_list_items)}: {id} - {title}")

        # Generate explanation
        explanation = generate_explanation(title, max_tokens=120)

        # Generate image prompt
        img_prompt = generate_image_prompt(title, explanation, max_tokens=69)

        # Append processed item
        processed_list.append({
            "id": id,
            "title": title,
            "explanation": explanation,
            "img_prompt": img_prompt
        })

    # Step 3: Combine everything into the desired output format
    output_data = {
        "project_name": project_name,
        "plan": plan_text,
        "numbered_list": processed_list
    }

    # Save the final output to a JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Final output saved to {output_file}")

    # ---- PRINT AND SAVE FORMATTED OUTPUT TO TXT ----
    txt_output_lines = []

    # Print plan heading and text
    print("\nPLAN")
    print(plan_text)
    txt_output_lines.append("PLAN")
    txt_output_lines.append(plan_text)
    txt_output_lines.append("")  # Blank line

    # Helper to determine depth from id (e.g., "1", "1.1", "1.1.1")
    def get_depth(id_str):
        return id_str.count(".")

    def get_dash(depth):
        if depth == 0:
            return '-' * 40
        elif depth == 1:
            return '-' * 20
        elif depth == 2:
            return '-' * 10
        else:
            return '-' * 5

    # Print topics and subtopics with improved dashes and correct order
    for item in processed_list:
        id = item["id"]
        title = item["title"]
        explanation = item["explanation"]
        img_prompt = item["img_prompt"]
        depth = get_depth(id)

        dash = get_dash(depth)

        # Add dash line above
        print(dash)
        txt_output_lines.append(dash)

        # Add id and title
        print(f"{id}. {title}")
        txt_output_lines.append(f"{id}. {title}")

        # Add explanation
        print(f"Explanation: {explanation}")
        txt_output_lines.append(f"Explanation: {explanation}")

        # Add image prompt
        print(f"Image Prompt: {img_prompt}")
        txt_output_lines.append(f"Image Prompt: {img_prompt}")

        txt_output_lines.append("")  # Blank line

    # Save to txt file
    txt_file = output_file.rsplit(".", 1)[0] + ".txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_output_lines))

    print(f"\nFormatted output also saved to {txt_file}")


if __name__ == "__main__":
    main()