import os
import glob
from bs4 import BeautifulSoup

# Define directories.
INPUT_DIR = "RAG_content/Website"
OUTPUT_DIR = "RAG_content/removedHTMLText"

def process_html_file(input_path, output_path):
    # Read the HTML file.
    with open(input_path, "r", encoding="utf8") as f:
        html_content = f.read()

    # Parse with BeautifulSoup.
    soup = BeautifulSoup(html_content, "html.parser")

    # Process all <a> tags: if the link starts with http(s), append the URL.
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and href.startswith("http"):
            # Insert the URL after the anchor text, in parentheses.
            # This helps preserve links in the plain text.
            a.insert_after(" (" + href + ")")
            # Replace the entire <a> tag with just its text.
            a.replace_with(a.get_text())

    # Extract the text from the soup.
    text = soup.get_text(separator="\n")
    
    # Write the text content to the output path.
    with open(output_path, "w", encoding="utf8") as f:
        f.write(text)

def main():
    # Create the output directory if it doesn't exist.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Recursively get all HTML files in the input directory.
    html_files = glob.glob(os.path.join(INPUT_DIR, '**', '*.html'), recursive=True)

    for html_file in html_files:
        # Compute the relative path from the input directory.
        rel_path = os.path.relpath(html_file, INPUT_DIR)
        # Build a corresponding output path with .txt extension.
        output_file = os.path.join(OUTPUT_DIR, os.path.splitext(rel_path)[0] + ".txt")
        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            process_html_file(html_file, output_file)
            print(f"Processed: {html_file} -> {output_file}")
        except Exception as e:
            print(f"Error processing {html_file}: {e}")

if __name__ == "__main__":
    main()
