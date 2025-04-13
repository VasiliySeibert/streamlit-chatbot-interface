import json
import time
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Load your existing JSON
with open("nfdi4ing_works_extended.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

# Simple link check
def is_valid_link(link):
    return isinstance(link, str) and link.strip().startswith("http")

# Process entries with progress bar
for entry in tqdm(entries, desc="Fetching abstracts"):
    doi_link = entry.get("link", "").strip()

    if not is_valid_link(doi_link):
        entry["abstract"] = "No valid DOI/link"
        continue

    try:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a research assistant. you will receive a doi and your job is to "
                                "conduct a websearch and try to retrieve the abstract. If you don't find an "
                                "abstract, just say so, don’t make anything up. If you find one repeat it word for word exactly, without altering its contents"
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": doi_link
                        }
                    ]
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "medium"
                }
            ],
            temperature=1,
            max_output_tokens=2048,
            top_p=1,
            store=True
        )

        # Corrected access to the abstract text
        abstract = response.output[1].content[0].text.strip()
        entry["abstract"] = abstract

    except Exception as e:
        entry["abstract"] = f"Error: {str(e)}"

    

    # time.sleep(1)

# Save result
with open("nfdi4ing_works_with_abstracts_partial.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print("✅ Saved: nfdi4ing_works_with_abstracts_partial.json")
