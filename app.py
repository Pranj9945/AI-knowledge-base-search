from flask import Flask, request, jsonify, send_from_directory, render_template_string, redirect, url_for
import os, uuid, shutil, io
from retriever import Retriever
from answerer import extract_candidate_sentences, simulate_llm_generate
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = 'docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# prefer embeddings if available
retriever = Retriever(docs_folder=UPLOAD_FOLDER, use_embeddings=True)

# Try to use pdfplumber and python-docx for parsing uploaded files into text if present
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None

INDEX_HTML = '''<!doctype html>
<html>
<head><title>RAG KB Prototype v2</title></head>
<body style="font-family: Arial, sans-serif; margin:40px;">
<h2>AI Knowledge Base — Upload & Query (Prototype v2)</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
  <label>Upload document (.txt, .pdf, .docx):</label><br/>
  <input type="file" name="file"/><br/><br/>
  <input type="submit" value="Upload"/>
</form>
<hr/>
<form id="qform" action="/ask" method="post">
  <label>Ask a question:</label><br/>
  <input style="width:60%" type="text" name="question" placeholder="Enter your question"/><br/><br/>
  <label>Top-k retrieved docs:</label>
  <input type="number" name="top_k" value="3" min="1" max="10"/>
  <input type="submit" value="Ask"/>
</form>
<hr/>
<div id="result">
{% if result %}
  <h3>Answer</h3>
  <pre>{{ result | tojson(indent=2) }}</pre>
{% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, result=None)

def parse_file_to_text(file_stream, filename):
    # .txt: return text
    fname = filename.lower()
    if fname.endswith('.txt'):
        try:
            return file_stream.read().decode('utf-8')
        except:
            try:
                return file_stream.read().decode('latin-1')
            except:
                return ''
    if fname.endswith('.pdf') and pdfplumber:
        try:
            # write to temp and read via pdfplumber
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tmp.write(file_stream.read())
            tmp.close()
            text = []
            with pdfplumber.open(tmp.name) as pdf:
                for p in pdf.pages:
                    text.append(p.extract_text() or '')
            os.unlink(tmp.name)
            return '\n'.join(text)
        except Exception as e:
            print('PDF parse error', e)
            return ''
    if fname.endswith('.docx') and docx:
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
            tmp.write(file_stream.read())
            tmp.close()
            doc = docx.Document(tmp.name)
            full = '\n'.join(p.text for p in doc.paragraphs)
            os.unlink(tmp.name)
            return full
        except Exception as e:
            print('DOCX parse error', e)
            return ''
    return ''

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return 'No file part', 400
        f = request.files['file']
        if f.filename == '':
            return 'No selected file', 400
        # save original file
        fname = f.filename
        save_path = os.path.join(UPLOAD_FOLDER, fname)
        f.seek(0)
        content = f.read()
        with open(save_path, 'wb') as out:
            out.write(content)
        # also create a .txt representation for retrieval if parsing available
        text = parse_file_to_text(io.BytesIO(content), fname)
        txt_path = os.path.splitext(save_path)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as t:
            t.write(text)
        retriever.refresh()
        return redirect(url_for('index'))
    except Exception as e:
        traceback.print_exc()
        return str(e), 500

@app.route('/ask', methods=['POST'])
def ask_web():
    q = request.form.get('question')
    try:
        top_k = int(request.form.get('top_k', 3))
    except:
        top_k = 3
    results = retriever.retrieve(q, top_k=top_k)
    candidates = extract_candidate_sentences(results, q, max_sentences=6)
    answer, confidence, sources, missing, suggestions = simulate_llm_generate(q, candidates, top_k_sent=3)
    resp = {
        'answer': answer,
        'confidence': confidence,
        'sources': sources,
        'missing_info': missing,
        'enrichment_suggestions': suggestions
    }
    return render_template_string(INDEX_HTML, result=resp)

@app.route('/query', methods=['POST'])
def query_api():
    data = request.get_json() or {}
    q = data.get('question') or data.get('query')
    if not q:
        return jsonify({'error':'provide question in JSON as {"question":"..."}'}), 400
    top_k = int(data.get('top_k', 3))
    results = retriever.retrieve(q, top_k=top_k)
    candidates = extract_candidate_sentences(results, q, max_sentences=6)
    answer, confidence, sources, missing, suggestions = simulate_llm_generate(q, candidates, top_k_sent=3)
    resp = {
        'answer': answer,
        'confidence': confidence,
        'sources': sources,
        'missing_info': missing,
        'enrichment_suggestions': suggestions
    }
    return jsonify(resp)

if __name__ == '__main__':
    print('Starting app. Open http://127.0.0.1:5000 in your browser.')
    app.run(debug=True)
