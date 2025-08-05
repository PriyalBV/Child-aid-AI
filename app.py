from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import torch
import os
import re

app = Flask(__name__)

# --- Global Configuration ---
DOCUMENT_FOLDER = 'documents'
documents = []
tags = []
titles = []

# --- Step 1: Load Documents and Tags ---
for filename in os.listdir(DOCUMENT_FOLDER):
    if filename.endswith('.txt'):
        with open(os.path.join(DOCUMENT_FOLDER, filename), 'r') as file:
            lines = file.readlines()

            # Extract tags
            tag_line = next((l for l in lines if l.lower().startswith('tags:')), '')
            tag_content = tag_line[len('Tags:'):].strip().lower()
            tags.append(tag_content)

            # Remove the tag line from content
            content_lines = [l for l in lines if not l.lower().startswith('tags:')]
            content = ''.join(content_lines)
            documents.append(content)

            # Save the title (filename without .txt)
            title_line = next((l for l in lines if l.lower().startswith('title:')), '')
            title = title_line[len('Title:'):].strip()
            titles.append(title if title else filename.replace('.txt', ''))

# --- Step 2: Load Sentence Transformer Model ---
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# --- Step 3: Helper Functions ---

def extract_age_from_query(query):
    """Extract age from a query string if present."""
    match = re.search(r'\b(\d{1,2})\b', query)
    return int(match.group(1)) if match else None

def filter_documents_by_keywords(query, docs, tags):
    """Filter documents based on age range or keyword match in tags."""
    query_words = query.lower().split()
    query_age = extract_age_from_query(query)
    
    filtered_docs = []
    filtered_indices = []

    for i, tag_string in enumerate(tags):
        tag_words = tag_string.split(',')

        age_match = False
        if query_age is not None:
            for tag in tag_words:
                if 'age:' in tag:
                    range_part = tag.split(':')[1].strip()
                    if '-' in range_part:
                        try:
                            start, end = map(int, range_part.split('-'))
                            if start <= query_age <= end:
                                age_match = True
                        except ValueError:
                            continue  # Skip malformed tags

        keyword_match = any(word in tag_string for word in query_words)

        if keyword_match or age_match:
            filtered_docs.append(docs[i])
            filtered_indices.append(i)

    return filtered_docs, filtered_indices

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Step 4: Filter relevant documents
    filtered_docs, indices = filter_documents_by_keywords(query, documents, tags)

    if not filtered_docs:
        return render_template('result.html', query=query, results=[{
            'title': 'No relevant matches found',
            'content': 'Try rephrasing your query or using different terms.',
            'score': 0
        }])

    # Step 5: Semantic ranking using sentence embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(filtered_docs, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_k = torch.topk(similarities, k=min(3, len(filtered_docs)))

    results = []
    for idx, score in zip(top_k.indices, top_k.values):
        original_index = indices[idx]
        results.append({
            'title': titles[original_index],
            'content': documents[original_index],
            'score': round(float(score), 4)
        })

    return render_template('result.html', query=query, results=results)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
