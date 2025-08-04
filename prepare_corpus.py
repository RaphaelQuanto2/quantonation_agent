import os
import re
import docx
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
SOURCE_DIR = "/Users/raphael/Documents/Quantonation's Agent/Documents for Training"
OUTPUT_FILE = "processed_chunks.jsonl"
CHUNK_SIZE = 800  # tokens or words (approx)
OVERLAP = 200     # overlap between chunks

# --- TEXT EXTRACTORS ---
def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

# --- CHUNKING ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:  # skip very short ones
            chunks.append(chunk)
    return chunks

# --- MAIN ---
def process_documents():
    import json
    out = open(OUTPUT_FILE, "w", encoding="utf-8")

    files = list(Path(SOURCE_DIR).rglob("*.*"))
    for file_path in tqdm(files, desc="Processing documents"):
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                text = extract_pdf(file_path)
            elif ext == ".docx":
                text = extract_docx(file_path)
            elif ext == ".txt":
                text = extract_txt(file_path)
            else:
                continue

            text = re.sub(r"\s+", " ", text).strip()
            if not text or len(text) < 500:
                continue

            chunks = chunk_text(text)
            for chunk in chunks:
                record = {
                    "source": str(file_path),
                    "content": chunk
                }
                out.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")
    out.close()

if __name__ == "__main__":
    process_documents()
