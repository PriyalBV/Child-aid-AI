from django.shortcuts import render
from .utils import filter_docs, model
from sentence_transformers import util

def home(request):
    return render(request, "index.html")


def search(request):
    if request.method == "POST":
        query = request.POST.get("query")
        matched_docs, model_instance = filter_docs(query)

        if not matched_docs:
            return render(
                request,
                "result.html",
                {
                    "query": query,
                    "results": [
                        {
                            "title": "No matches found",
                            "content": "Try using different keywords.",
                            "score": 0,
                        }
                    ],
                },
            )

        query_embedding = model_instance.encode(query, convert_to_tensor=True)
        scored_results = []

        for doc in matched_docs:
            with open(doc["path"], "r") as f:
                lines = f.readlines()
            content = "".join(
                [l for l in lines if not l.lower().startswith(("tags:", "title:"))]
            )
            doc_embedding = model_instance.encode(content, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            scored_results.append(
                {"title": doc["title"], "content": content, "score": round(score, 4)}
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_results[:3]

        return render(request, "result.html", {"query": query, "results": top_results})

    return render(request, "index.html")
