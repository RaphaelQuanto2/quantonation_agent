import os
import json
import re
import datetime
import requests
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from typing import List
import numpy as np
import faiss
import streamlit as st
import openai
import os

# --- Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("\U0001F4C2 Working directory set to:", os.getcwd())

NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
openai.api_key = OPENAI_API_KEY

if not OPENAI_API_KEY or not NOTION_TOKEN or not NOTION_DATABASE_ID:
    raise EnvironmentError("❌ Missing env variables")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

notion_headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # Consistent with FAISS setup

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def inject_pdf_into_faiss(uploaded_files, index, corpus_texts):
    new_chunks = []
    for uploaded_file in uploaded_files:
        pdf_text = extract_text_from_pdf(uploaded_file)
        for chunk in pdf_text.split("\n\n"):
            cleaned = chunk.strip()
            if len(cleaned) > 50:
                corpus_texts.append(cleaned)
                new_chunks.append(cleaned)

    if not new_chunks:
        return index, corpus_texts

    # Use OpenAI to embed (not sentence-transformers)
    resp = client.embeddings.create(model="text-embedding-3-small", input=new_chunks)
    new_embeddings = [e.embedding for e in resp.data]

    index.add(np.array(new_embeddings).astype("float32"))

    # Save index and texts
    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "w") as f:
        json.dump(corpus_texts, f)

    return index, corpus_texts


# --- Constants ---
CORPUS_PATH = "processed_chunks.jsonl"
INDEX_FILE = "faiss.index"
TEXTS_FILE = "corpus_texts.json"
EMBED_MODEL = "text-embedding-3-small"

# --- Utility functions ---
def truncate_words(text, limit=1999):
    return ' '.join(text.split()[:limit])

def extract_score(value_str):
    match = re.search(r'(\d+(\.\d+)?)', value_str)
    return float(match.group(1)) if match else None

def normalize_key(k):
    return re.sub(r'[^a-z0-9]', '', k.lower())

def parse_gpt_response(gpt_output):
    updates = {}
    lines = gpt_output.strip().split("\n")
    for line in lines:
        line = line.strip().lstrip("-•1234567890. ").strip()
        match = re.match(r"(.+?)\s*:\s*(.+)", line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            updates[key] = value
    print("\n✅ Parsed fields:", updates)
    return updates

def split_text_to_blocks(text, max_len=2000):
    blocks = []
    while text:
        chunk = text[:max_len]
        split_idx = chunk.rfind('\n')
        if split_idx != -1:
            chunk = text[:split_idx]
        blocks.append(chunk.strip())
        text = text[len(chunk):].lstrip()
    return blocks

def user_confirmation(prompt_msg):
    ans = input(f"{prompt_msg} [y/n]: ").strip().lower()
    return ans in ['y', 'yes']

# --- FAISS Setup ---
def build_faiss_index():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    texts = [doc["content"] for doc in corpus]
    print(f"🔍 Loaded {len(texts)} corpus chunks")

    embeddings = []
    for i in tqdm(range(0, len(texts), 100), desc="📐 Embedding corpus"):
        batch = texts[i:i+100]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([e.embedding for e in resp.data])

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "w") as f:
        json.dump(texts, f)

    return index, texts

def load_faiss_index():
    index = faiss.read_index(INDEX_FILE)
    with open(TEXTS_FILE) as f:
        texts = json.load(f)
    return index, texts

def search_corpus(index, corpus_texts, query, top_k=5):
    emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    D, I = index.search(np.array([emb]).astype("float32"), top_k)
    return [corpus_texts[i] for i in I[0]]

# --- Notion helpers ---
def update_problem_statement(page_id, text):
    payload = {
        "properties": {
            "Problem Statement": {
                "rich_text": [{"text": {"content": truncate_words(text, 1999)}}]
            }
        }
    }
    res = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=notion_headers, json=payload)
    print("✏️ Problem Statement updated." if res.status_code == 200 else f"❌ Failed: {res.text}")

def update_notion_properties(page_id, updates_dict):
    known_fields = {
        normalize_key("Technology Leveraged"): "Technology Leveraged",
        normalize_key("Market Size"): "Market Size",
        normalize_key("Competitive Advantage"): "Competitive Advantage",
        normalize_key("Feasibility Score (1–10)"): "Feasibility Score (1–10)",
        normalize_key("Investment Thesis Fit"): "Investment Thesis Fit",
        normalize_key("Next Steps"): "Next Steps",
        normalize_key("Problem Severity (1–10)"): "Problem Severity (1–10)",
        normalize_key("Tech Readiness Level"): "Tech Readiness Level",
        normalize_key("Tech Readiness Level (TRL 1–9)"): "Tech Readiness Level",
        normalize_key("Strategic Partner Ideas"): "Strategic Partner Ideas",
        normalize_key("Funding Needs"): "Funding Needs",
        normalize_key("Potential Founders / Talent"): "Potential Founders / Talent",
        normalize_key("Sector/Vertical"): "Sector/Vertical"
    }
    props = {}
    for k, value in updates_dict.items():
        field_key = normalize_key(k)
        field = known_fields.get(field_key)
        if not field:
            print(f"⚠️ Unknown or unmatched field: {k}")
            continue
        if any(x in field for x in ["Score", "Severity", "Level"]):
            num = extract_score(value)
            if num is not None:
                props[field] = {"number": num}
            elif value.lower() == "not specified":
                props[field] = {"number": None}
                print(f"ℹ️ Field '{field}' explicitly marked as not specified.")
            else:
                print(f"⚠️ Couldn't parse score for {field}: '{value}'")
        else:
            if value.lower() != "not specified":
                props[field] = {"rich_text": [{"text": {"content": truncate_words(value, 1999)}}]}

    if props:
        res = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=notion_headers, json={"properties": props})
        print("🛠 Updated:", list(props.keys()))
        print("🔄 Status:", res.status_code, res.text)

def create_notion_subpage(parent_id, title, markdown_text):
    chunks = split_text_to_blocks(markdown_text)
    children = [{
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": c}}]}
    } for c in chunks]

    payload = {
        "parent": {"type": "page_id", "page_id": parent_id},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": title}}]}
        },
        "children": children
    }
    res = requests.post("https://api.notion.com/v1/pages", headers=notion_headers, json=payload)
    print(f"📘 Subpage created: {title}" if res.status_code == 200 else f"❌ Subpage error: {res.text}")

# --- GPT logic ---
def generate_problem_statement(idea):
    prompt = f"""
You are a domain expert in deeptech. Formulate a rigorous and scientifically grounded problem statement underlying the following startup idea.

Startup idea: "{idea}"

The problem should:
– Be precise and technically specific
– Refer to known scientific or engineering limitations
– Highlight the unsolved nature of the challenge
– Be written in the tone of a grant application or academic abstract
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()[:1999]

def generate_gpt_output(idea, problem, snippets: List[str]):
    context = "\n\n".join(f"- {s}" for s in snippets)
    system_msg = """You are an expert in science and a Venture Capital Partner with years of experience. You're trying to assess key points for new ventures topics. Be specific, be precise, insightful. You can leverage the academic literature you know and your own research. 
Only respond in the following strict format, line by line, with no commentary and no bullet points:
Field: Value

Required fields:
Technology Leveraged
Market Size
Competitive Advantage
Feasibility Score (1–10)
Investment Thesis Fit
Next Steps
Problem Severity (1–10)
Tech Readiness Level (TRL 1–9)
Strategic Partner Ideas
Funding Needs
Potential Founders / Talent
Sector/Vertical
"""

    user_msg = f"""
Based on this scientific context:

{context}

Startup idea: "{idea}"
Problem: "{problem}"

Return each field exactly once. Do not add explanations.
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": truncate_words(user_msg)}
        ],
        temperature=0.2
    )
    output = resp.choices[0].message.content.strip()
    print("\n🎯 GPT Output Raw:\n", output)
    return output

def generate_deeptech_brief(idea, problem, snippets: List[str]):
    context = "\n\n".join(f"- {s}" for s in snippets)
    prompt = f"""
You are writing a deeptech opportunity memo, based on scientific context below and your own research. For Paragraph one you'll use the academic literature I provided, please do citations and index the referenced papers but do not reference twice the same paper, do not refer only to one paper but try to use as much as you can if it's relevant, when you cite paper please give information about the team that written this paper. Paragraph 2 and 3 are based on your own research, paragraph 3 should be a comprehensive list of existing companies in the field based. Paragraph 4 should be a list of relevant people to contact, based on both your research and literature. Structure your memo with the titles of each paragraph. 

{context}

Startup idea: "{idea}"
Problem: "{problem}"

Write a memo in 3 structured paragraphs:
1. Scientific context and state of the art
2. Market Analysis
3. Competitive Landscape
4. Relevant People

"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": truncate_words(prompt)}],
        temperature=0.6
    )
    return resp.choices[0].message.content.strip()

# --- Main ---
def run():
    if os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
        index, corpus_texts = load_faiss_index()
        print("✅ FAISS index loaded.")
    else:
        index, corpus_texts = build_faiss_index()
        print("🧠 FAISS index built and saved.")

    data = requests.post(f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query", headers=notion_headers).json()
    for item in data["results"]:
        page_id = item["id"]
        props = item["properties"]

        try:
            idea = props["Company Idea"]["title"][0]["text"]["content"]
        except:
            print("⛔ Skipping item with no title")
            continue

        try:
            problem = props["Problem Statement"]["rich_text"][0]["text"]["content"]
        except:
            print(f"✏️ Generating problem for: {idea}")
            problem = generate_problem_statement(idea)
            update_problem_statement(page_id, problem)

        print(f"\n💡 Enriching: {idea}")
        print(f"🔍 Problem: {problem}")

        if not user_confirmation("➡️ Proceed with enrichment?"):
            continue

        context_snippets = search_corpus(index, corpus_texts, f"{idea}. {problem}", top_k=5)
        gpt_resp = generate_gpt_output(idea, problem, context_snippets)
        updates = parse_gpt_response(gpt_resp)
        update_notion_properties(page_id, updates)

        memo = generate_deeptech_brief(idea, problem, context_snippets)
        title = f"{datetime.datetime.now().strftime('%Y-%m-%d')} – Memo: {idea[:60]}"
        create_notion_subpage(page_id, title, memo)

if __name__ == "__main__":
    run()
