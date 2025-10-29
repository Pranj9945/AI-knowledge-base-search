# AI-Powered Knowledge Base Search & Enrichment (Prototype v2)

This enhanced prototype implements the Challenge 2 requirements with additional features and a simple web UI (Option A).  
It is designed to run **offline** and **simulate** LLM behavior so no API keys or internet access are required.

## Highlights (v2)
- **Semantic retrieval**: uses sentence-transformers if installed; otherwise falls back to TF-IDF.
- **Simulated LLM**: local generator that composes an answer from retrieved passages and reformulates it.
- **Auto-enrichment**: when the answer lacks specific facts (dates, names, numbers), the system suggests documents/actions.
- **File support**: upload `.txt`, `.pdf`, `.docx` (parsing requires optional libraries).
- **Simple Web UI**: single-page HTML form for upload and query (served by Flask).
- **Structured JSON output**: `answer`, `confidence`, `sources`, `missing_info`, `enrichment_suggestions`.
- **Tests**: `tests.py` demonstrates all features with sample documents.

## Quick start (recommended)
1. Create and activate a virtual environment (Python 3.9+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows PowerShell
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Optional (for better retrieval & parsing):
   ```bash
   pip install sentence-transformers pdfplumber python-docx
   ```
   If you cannot install `sentence-transformers`, the app will use a TF-IDF fallback.
3. (Optional) If you want better sentence splitting, download NLTK punkt:
   ```py
   import nltk; nltk.download('punkt')
   ```
4. Run the demo tests:
   ```bash
   python tests.py
   ```
5. Run the web app (serves UI at http://127.0.0.1:5000):
   ```bash
   python app.py
   ```

## Usage (Web UI)
- Open `http://127.0.0.1:5000` in your browser.
- Use the **Upload Document** form to upload `.txt`, `.pdf`, or `.docx` files (PDF/DOCX require extra libs).
- Use the **Ask Question** form to send a question; you will receive a structured JSON-like answer and a readable text answer.

## API endpoints
- `POST /upload` - form file field `file` (.txt/.pdf/.docx)
- `POST /query` - JSON body `{"question":"...", "top_k":3}`

## Design notes
- **Semantic retrieval**: If `sentence-transformers` is installed, embeddings are computed and cosine similarity is used to retrieve top-k documents and passages. This yields better semantic matches. If it's not available, a TF-IDF vectorizer is used as a fallback.
- **Simulated LLM**: A local generator stitches retrieved sentences and runs simple level-of-detail heuristics to produce a fluent answer and a confidence score. This is deterministic and safe for offline use.
- **Auto-enrichment**: Heuristics detect missing specific facts (dates, names, numbers). The system returns `missing_info` and `enrichment_suggestions` with actionable items (e.g., "Add timeline doc", "Add bios").

## Where to improve
- Replace the simulated LLM with a real LLM for better language quality (requires API & costs).
- Use a database and chunking for large corpora, plus vector DB like FAISS for speed.
- Add user feedback loop and rating UI per the challenge stretch goals.

---
If you want further enhancements (React UI, LLM integration, auto-enrichment fetching from the web), tell me and I will implement them.
