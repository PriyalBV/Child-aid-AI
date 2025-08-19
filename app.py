from flask import Flask, render_template, request
import os
import re
import requests
import json

app = Flask(__name__)
DOCUMENT_FOLDER = "documents"

# --- Hugging Face API setup ---
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/paraphrase-MiniLM-L3-v2"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Set your Hugging Face token in Replit Secrets

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

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

# --- Helper: Extract age ---
def extract_age_from_query(query):
    match = re.search(r"\b(\d{1,2})\b", query)
    return int(match.group(1)) if match else None

# --- Helper: Filter relevant docs ---
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

    return matched

# --- Helper: Hugging Face embedding ---
def get_embedding(text):
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        embedding = response.json()[0]  # Extract vector
        return embedding
    else:
        print("HF API error:", response.status_code, response.text)
        return None

# --- Helper: Cosine similarity ---
def cosine_similarity(vec1, vec2):
    import numpy as np
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    matched_docs = filter_docs(query)

    if not matched_docs:
        return render_template(
            "result.html",
            query=query,
            results=[
                {
                    "title": "No matches found",
                    "content": "Try using different keywords.",
                    "score": 0,
                }
            ],
        )

    query_embedding = get_embedding(query)
    if not query_embedding:
        return "Error fetching embedding from Hugging Face API"

    scored_results = []

    for doc in matched_docs:
        with open(doc["path"], "r") as f:
            lines = f.readlines()
        content = "".join(
            [l for l in lines if not l.lower().startswith(("tags:", "title:"))]
        )
        doc_embedding = get_embedding(content)
        if not doc_embedding:
            continue
        score = cosine_similarity(query_embedding, doc_embedding)
        scored_results.append(
            {"title": doc["title"], "content": content, "score": round(score, 4)}
        )

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:3]

    return render_template("result.html", query=query, results=top_results)

# --- Deploy Ready ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
