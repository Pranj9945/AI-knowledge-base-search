# Demo tests for enhanced prototype v2
from retriever import Retriever
from answerer import extract_candidate_sentences, simulate_llm_generate
import os, json

def pretty(o):
    print(json.dumps(o, indent=2, ensure_ascii=False))

def run_tests():
    r = Retriever(docs_folder='docs', use_embeddings=True)
    queries = [
        'What is Product X and when was it launched?',
        'Who are the key team members of Project Alpha?',
        'How do I install the system locally?',
        'What is the return policy?',
        'How many users signed up in 2023?'
    ]
    for q in queries:
        res = r.retrieve(q, top_k=3)
        cands = extract_candidate_sentences(res, q, max_sentences=6)
        answer, confidence, sources, missing, suggestions = simulate_llm_generate(q, cands, top_k_sent=3)
        out = {'question':q, 'answer':answer, 'confidence':confidence, 'sources':sources, 'missing_info':missing, 'enrichment_suggestions':suggestions}
        pretty(out)
        print('\n'+'-'*60+'\n')

if __name__=='__main__':
    print('Running enhanced demo tests (v2) — ensure sample docs are in docs/)')
    run_tests()
