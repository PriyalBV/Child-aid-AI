from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import torch
import os
import re

app = Flask(__name__)
DOCUMENT_FOLDER = 'documents'

# --- Load Model Once ---
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# --- Index Files with Metadata Only ---
file_index = []
for filename in os.listdir(DOCUMENT_FOLDER):
    if filename.endswith('.txt'):
        path = os.path.join(DOCUMENT_FOLDER, filename)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        tag_line = next((l for l in lines if l.lower().startswith('tags:')), '')
        title_line = next((l for l in lines if l.lower().startswith('title:')), '')

        file_index.append({
            'filename': filename,
            'path': path,
            'tags': tag_line[len('Tags:'):].strip().lower(),
            'title': title_line[len('Title:'):].strip() or filename.replace('.txt', '')
        })

# --- Helper: Extract age ---
def extract_age_from_query(query):
    match = re.search(r'\b(\d{1,2})\b', query)
    return int(match.group(1)) if match else None

# --- Helper: Filter relevant docs ---
def filter_docs(query):
    query_words = query.lower().split()
    query_age = extract_age_from_query(query)
    matched = []

    for doc in file_index:
        tag_str = doc['tags']
        tag_words = tag_str.split(',')

        age_match = False
        if query_age:
            for tag in tag_words:
                if 'age:' in tag:
                    try:
                        start, end = map(int, tag.split(':')[1].split('-'))
                        if start <= query_age <= end:
                            age_match = True
                    except: continue

        keyword_match = any(word in tag_str for word in query_words)

        if keyword_match or age_match:
            matched.append(doc)

    return matched

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    matched_docs = filter_docs(query)

    if not matched_docs:
        return render_template('result.html', query=query, results=[{
            'title': 'No matches found',
            'content': 'Try using different keywords.',
            'score': 0
        }])

    query_embedding = model.encode(query, convert_to_tensor=True)

    scored_results = []

    for doc in matched_docs:
        with open(doc['path'], 'r') as f:
            lines = f.readlines()
        content = ''.join([l for l in lines if not l.lower().startswith(('tags:', 'title:'))])
        doc_embedding = model.encode(content, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
        scored_results.append({
            'title': doc['title'],
            'content': content,
            'score': round(score, 4)
        })

    # Sort by score and take top 3
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    top_results = scored_results[:3]

    return render_template('result.html', query=query, results=top_results)

# --- Deploy Ready ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
