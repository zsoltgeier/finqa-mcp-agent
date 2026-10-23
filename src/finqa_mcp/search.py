"""Lexical retrieval over the current report text."""
import math
import re


def search_text(record, query, top_k=3):
    stop = {"the", "a", "an", "of", "in", "to", "for", "and", "was", "were", "what", "is", "as", "at", "by", "on", "from", "with", "how", "did"}
    terms = set(re.findall(r"[a-z0-9]+", query.lower())) - stop
    chunks = record.pre_text + record.post_text
    documents = [set(re.findall(r"[a-z0-9]+", text.lower())) - stop for text in chunks]
    idf = {term: math.log(1 + (len(documents) - sum(term in d for d in documents) + .5) /
                                    (sum(term in d for d in documents) + .5)) for term in terms}
    rows = []
    for side in ("pre", "post"):
        for i, value in enumerate(getattr(record, f"{side}_text")):
            words = set(re.findall(r"[a-z0-9]+", value.lower()))
            score = sum(idf[t] for t in terms & words) / max(1, len(words)) ** .5
            rows.append({"source_id": f"{side}_{i}", "text": value, "score": round(score, 6)})
    return {"example_id": record.id, "matches": sorted(rows, key=lambda x: -x["score"])[:top_k]}

