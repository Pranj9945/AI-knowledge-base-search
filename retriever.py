import os, json, math
from typing import List, Tuple
# Try to use sentence-transformers; otherwise fallback to TF-IDF
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Retriever:
    def __init__(self, docs_folder='docs', use_embeddings=True, model_name='all-MiniLM-L6-v2'):
        self.docs_folder = docs_folder
        self.use_embeddings = use_embeddings and _HAS_ST
        self.model_name = model_name
        self.docs = []  # list of (path, text)
        self.paths = []
        self.embeddings = None
        self.vectorizer = None
        self.doc_vectors = None
        if self.use_embeddings:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None
                self.use_embeddings = False
        else:
            self.model = None
        self._load_docs()

    def _load_docs(self):
        self.docs = []
        self.paths = []
        if not os.path.exists(self.docs_folder):
            os.makedirs(self.docs_folder)
        for fname in sorted(os.listdir(self.docs_folder)):
            if fname.lower().endswith(('.txt','.pdf','.docx')):
                path = os.path.join(self.docs_folder, fname)
                with open(path, 'rb') as f:
                    try:
                        txt = f.read().decode('utf-8')
                    except Exception:
                        # fallback: treat as empty string; parsing done elsewhere
                        txt = ''
                self.docs.append(txt)
                self.paths.append(path)
        if self.use_embeddings and self.model and self.docs:
            # compute embeddings
            self.embeddings = self.model.encode(self.docs, convert_to_tensor=True)
        elif self.docs:
            # TF-IDF fallback on raw texts
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
            self.doc_vectors = self.vectorizer.fit_transform(self.docs)

    def refresh(self):
        self._load_docs()

    def retrieve(self, query:str, top_k:int=3) -> List[Tuple[str,str,float]]:
        """Return list of (path, text, score)"""
        if self.use_embeddings and self.model and self.embeddings is not None:
            q_emb = self.model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, self.embeddings).cpu().numpy().ravel()
            idx = np.argsort(-sims)[:top_k]
            results = []
            for i in idx:
                if sims[i] <= 0:
                    continue
                results.append((self.paths[i], self.docs[i], float(sims[i])))
            return results
        elif self.doc_vectors is not None:
            qv = self.vectorizer.transform([query])
            sims = (self.doc_vectors @ qv.T).toarray().ravel()
            idx = np.argsort(-sims)[:top_k]
            results = []
            for i in idx:
                if sims[i] <= 0:
                    continue
                results.append((self.paths[i], self.docs[i], float(sims[i])))
            return results
        else:
            return []
