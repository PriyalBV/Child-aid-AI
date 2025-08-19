# main/utils.py
from sentence_transformers import SentenceTransformer, util
import os, re

DOCUMENT_FOLDER = "documents"

# --- Load Model Once ---
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# --- Index Files with Metadata Only ---
file_index = []
for filename in os.listdir(DOCUMENT_FOLDER):
    if filename.endswith(".txt"):
        path = os.path.join(DOCUMENT_FOLDER, filename)
        with open(path, "r") as f:
            lines = f.readlines()

        tag_line = next((l for l in lines if l.lower().startswith("tags:")), "")
        title_line = next((l for l in lines if l.lower().startswith("title:")), "")

        file_index.append(
            {
                "filename": filename,
                "path": path,
                "tags": tag_line[len("Tags:") :].strip().lower(),
                "title": title_line[len("Title:") :].strip()
                or filename.replace(".txt", ""),
            }
        )


def extract_age_from_query(query):
    match = re.search(r"\b(\d{1,2})\b", query)
    return int(match.group(1)) if match else None


def filter_docs(query):
    query_words = query.lower().split()
    query_age = extract_age_from_query(query)
    matched = []

    for doc in file_index:
        tag_str = doc["tags"]
        tag_words = tag_str.split(",")

        age_match = False
        if query_age:
            for tag in tag_words:
                if "age:" in tag:
                    try:
                        start, end = map(int, tag.split(":")[1].split("-"))
                        if start <= query_age <= end:
                            age_match = True
                    except:
                        continue

        keyword_match = any(word in tag_str for word in query_words)

        if keyword_match or age_match:
            matched.append(doc)

    return matched, model
