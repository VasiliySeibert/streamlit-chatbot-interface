from openai import OpenAI
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

input_file = "nfdi4ing_links.txt"
output_file = "website_analysis_results.json"
error_log_file = "website_analysis_errors.txt"
max_retries = 3
delay = 2  # seconds between calls
max_to_process = 20  # stop after 3 URLs for testing

# Load all URLs from input file
with open(input_file, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

# Load already processed results
results = []
processed_urls = set()

if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
            processed_urls = {entry["url"].rstrip("/") for entry in results}
        except json.JSONDecodeError:
            print("⚠️ Failed to parse existing results file. Starting fresh.")
            results = []
            processed_urls = set()

# Keep track of failed URLs
failed_urls = []
processed_count = 0

# Process URLs
for url in tqdm(urls, desc="Processing URLs", unit="url"):
    url_normalized = url.rstrip("/")
    if url_normalized in processed_urls:
        continue

    if processed_count >= max_to_process:
        break

    retries = 0
    while retries < max_retries:
        try:
            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Visit the following website and analyze its content:\n{url}\n\nProvide a structured JSON output containing:\n\nThe page title and a short summary.\n\nA list of 5 key takeaways, each with a 1000-character explanation.\n\nA list of follow-up search queries that would help deepen understanding of the topic."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": url
                            }
                        ]
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "website_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "key_takeaways": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "detail": {"type": "string"}
                                        },
                                        "required": ["title", "detail"],
                                        "additionalProperties": False
                                    }
                                },
                                "suggested_search_queries": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["url", "title", "summary", "key_takeaways", "suggested_search_queries"],
                            "additionalProperties": False
                        }
                    }
                },
                reasoning={},
                tools=[{
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "medium"
                }],
                temperature=1,
                max_output_tokens=2048,
                top_p=1,
                store=True
            )

            # Extract and parse result
            json_string = response.output[1].content[0].text
            print("\n🧠 Raw JSON string:\n", json_string)

            parsed_result = json.loads(json_string)
            results.append(parsed_result)

            # Save updated results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            processed_count += 1
            time.sleep(delay)
            break

        except Exception as e:
            retries += 1
            print(f"\n⚠️ Error processing {url} (attempt {retries}): {e}")
            time.sleep(delay * retries)

    if retries == max_retries:
        failed_urls.append(url)
        with open(error_log_file, "a", encoding="utf-8") as f:
            f.write(f"{url}\n")

# Final report
print(f"\n✅ Done! Processed {processed_count} new URL(s) successfully.")
if failed_urls:
    print(f"❌ {len(failed_urls)} URL(s) failed. See {error_log_file}.")
