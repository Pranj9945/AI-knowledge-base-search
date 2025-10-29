import re, random, math
from nltk import sent_tokenize
from typing import List, Tuple

def simple_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split('.') if s.strip()]

def extract_candidate_sentences(retrieved_docs, query, max_sentences=6):
    q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t)>1]
    candidates = []
    for path, text, score in retrieved_docs:
        sents = simple_sent_tokenize(text)
        for s in sents:
            s_clean = s.strip()
            if not s_clean: continue
            # simple relevance: token overlap
            s_tokens = [t.lower() for t in re.findall(r"\w+", s_clean)]
            overlap = sum(1 for t in q_tokens if t in s_tokens)
            if overlap>0:
                candidates.append((overlap, score, path, s_clean))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    # return top few
    return candidates[:max_sentences]

def simulate_llm_generate(query:str, candidates:List[Tuple], top_k_sent=3):
    """Simulate a local LLM by stitching top candidate sentences and applying simple paraphrase rules."""
    if not candidates:
        # No info
        return ("I could not find relevant information in the knowledge base for this question.", 0.15, [], ["No matching content in KB for query tokens"], [f"Upload documents about: '{query}'"])
    # pick top sentences
    chosen = candidates[:top_k_sent]
    sentences = [c[3] for c in chosen]
    sources = list(dict.fromkeys([c[2] for c in chosen]))
    # simple paraphrase: combine and add connective phrases
    intro = f"Based on the knowledge base, here's what I can tell you about '{query}':"
    body = ' '.join(sentences)
    # sometimes add clarifying sentence
    clarifier = ''
    if len(sentences)>1:
        clarifier = ' If you want more details, ask a follow-up question.'
    answer = intro + ' ' + body + clarifier
    # confidence heuristic
    avg_overlap = sum(c[0] for c in chosen)/len(chosen)
    avg_score = sum(c[1] for c in chosen)/len(chosen)
    confidence = min(0.98, 0.15 + 0.4*(avg_overlap) + 0.4*(avg_score))
    # completeness checks
    missing = []
    q_lower = query.lower()
    # dates
    if any(w in q_lower for w in ['when','date','year','launche','launched']) and not re.search(r'\b(19|20)\d{2}\b', body):
        missing.append('Missing specific dates or years')
    # persons
    if any(w in q_lower for w in ['who','name','person','team','members']) and not re.search(r'[A-Z][a-z]+\s[A-Z][a-z]+', body):
        missing.append('Missing named persons or roles')
    # numbers/quantities
    if any(w in q_lower for w in ['how many','number','count']) and not re.search(r'\b\d+\b', body):
        missing.append('Missing numeric data or counts')
    # enrichment suggestions
    suggestions = []
    if 'dates' in ' '.join(missing).lower():
        suggestions.append('Add timeline or release notes documents containing dates and versions')
    if 'persons' in ' '.join(missing).lower():
        suggestions.append('Add team bios, organizational charts, or contributor lists')
    if 'numeric' in ' '.join(missing).lower() or not suggestions:
        suggestions.append('Add more detailed spec sheets, analytics exports, or policy documents related to the question')
    return (answer, float(confidence), sources, missing, suggestions)
