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
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

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

# --- Embedding model for FAISS ---
model = SentenceTransformer("all-MiniLM-L6-v2")  # Consistent with FAISS setup

# --- Constants ---
CORPUS_PATH = "processed_chunks.jsonl"
INDEX_FILE = "faiss.index"
TEXTS_FILE = "corpus_texts.json"
EMBED_MODEL = "text-embedding-3-small"


# --- Utility functions ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


def embed_chunks_with_openai(chunks, model=EMBED_MODEL, batch_size=80):
    """Safely embeds text chunks using OpenAI API, in batches."""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = [c.strip()[:5000] for c in chunks[i:i + batch_size] if isinstance(c, str) and c.strip()]
        if not batch:
            continue
        try:
            resp = client.embeddings.create(model=model, input=batch)
            embeddings.extend([e.embedding for e in resp.data])
        except Exception as e:
            print("❌ Embedding API error:", e)
            raise e
    return embeddings


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

    # ✅ Safely batch and embed with OpenAI
    new_embeddings = embed_chunks_with_openai(new_chunks, model=EMBED_MODEL)

    # ✅ Add to FAISS
    index.add(np.array(new_embeddings).astype("float32"))

    # ✅ Save to disk
    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "w") as f:
        json.dump(corpus_texts, f)

    return index, corpus_texts


def truncate_words(text, limit=1999):
    return " ".join(text.split()[:limit])


def extract_score(value_str):
    match = re.search(r"(\d+(\.\d+)?)", value_str)
    return float(match.group(1)) if match else None


def normalize_key(k):
    return re.sub(r"[^a-z0-9]", "", k.lower())


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
        split_idx = chunk.rfind("\n")
        if split_idx != -1:
            chunk = text[:split_idx]
        blocks.append(chunk.strip())
        text = text[len(chunk):].lstrip()
    return blocks


def user_confirmation(prompt_msg):
    ans = input(f"{prompt_msg} [y/n]: ").strip().lower()
    return ans in ["y", "yes"]


# --- FAISS Setup ---
def build_faiss_index():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    texts = [doc["content"] for doc in corpus]
    print(f"🔍 Loaded {len(texts)} corpus chunks")

    embeddings = []
    for i in tqdm(range(0, len(texts), 100), desc="📐 Embedding corpus"):
        batch = texts[i:i + 100]
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
    res = requests.patch(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=notion_headers,
        json=payload,
    )
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
        normalize_key("Sector/Vertical"): "Sector/Vertical",
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
                props[field] = {
                    "rich_text": [
                        {"text": {"content": truncate_words(value, 1999)}}
                    ]
                }

    if props:
        res = requests.patch(
            f"https://api.notion.com/v1/pages/{page_id}",
            headers=notion_headers,
            json={"properties": props},
        )
        print("🛠 Updated:", list(props.keys()))
        print("🔄 Status:", res.status_code, res.text)


def create_notion_subpage(parent_id, title, markdown_text):
    chunks = split_text_to_blocks(markdown_text)
    children = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": c}}]
            },
        }
        for c in chunks
    ]

    payload = {
        "parent": {"type": "page_id", "page_id": parent_id},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": title}}]}
        },
        "children": children,
    }
    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=notion_headers,
        json=payload,
    )
    print(
        f"📘 Subpage created: {title}"
        if res.status_code == 200
        else f"❌ Subpage error: {res.text}"
    )


# --- GPT logic: PROBLEM STATEMENT (company / pitch style) ---
def generate_problem_statement(idea: str) -> str:
    prompt = f"""
You are a deeptech VC partner and former operator.

Write a **company-level problem statement** for the following startup idea, as it would appear on the *Problem* slide of a seed-stage pitch deck.

Startup idea: "{idea}"

Requirements for the problem statement:
- 2–4 sentences, no bullets.
- Explicitly state **who** has the problem (customer segment, typical role, industry).
- Describe the current workflow / status quo and why it is inadequate.
- Quantify the pain with order-of-magnitude metrics (e.g. time wasted, error rates, costs, yield loss, capex/opex, risk).
- Anchor it in real-world constraints (scientific, engineering, regulatory, or infrastructure limitations).
- Make the problem urgent and recurring, not a nice-to-have.
- Use clear, non-hyped language suitable for a professional investor memo.

Output only the final problem statement paragraph. No headings, no preamble.
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()[:1999]


# --- GPT logic: STRUCTURED FIELDS (VC memo style) ---
def generate_gpt_output(idea: str, problem: str, snippets: List[str]) -> str:
    context = "\n\n".join(f"- {s}" for s in snippets)

    system_msg = """
You are a senior deeptech VC partner at Quantonation, writing internal notes for a potential investment.

Goal: Fill in **concise, pitch-quality** fields that could be copied directly into a one-page investment memo for THIS specific startup idea.

Style:
- Be precise, concrete, and non-generic.
- Prefer numbers, examples, and named subsectors over buzzwords.
- Write each value as 1–3 short sentences on a single line (no line breaks).
- Do NOT invent unrealistic numbers; use sensible order-of-magnitude estimates.

Only respond in the following strict format, line by line, with no commentary and no bullet points:
Field: Value

Required fields (content expectations):
- Technology Leveraged → Core scientific/engineering approach (e.g. “trapped-ion quantum computing with photonic interconnects”, “III-V photonic integrated circuits for mid-IR sensing”) and why it matters for this use case.
- Market Size → Quantified view with at least one geography and segment: TAM / SAM / initial wedge, with indicative $ values and a sentence on adoption driver.
- Competitive Advantage → 2–3 crisp differentiators separated by semicolons (“…; …; …”), focusing on what is hard to copy (IP, data, infrastructure, regulatory position, ecosystem).
- Feasibility Score (1–10) → “X/10 – short justification about technical and execution risk”.
- Investment Thesis Fit → 1–2 sentences on why this fits a deeptech / Quantonation-type thesis (technical depth, defensibility, timing).
- Next Steps → 2–4 concrete milestones separated by semicolons (e.g. “validate X with 2 design partners; hire founding ML engineer; file priority patents; build go-to-market advisor bench”).
- Problem Severity (1–10) → “X/10 – short justification referencing economic/strategic pain”.
- Tech Readiness Level (TRL 1–9) → “X/9 – justification citing prototype stage, lab results, or pilot status”.
- Strategic Partner Ideas → Names/types of specific potential partners (corporates, labs, integrators, consortia) and what they bring.
- Funding Needs → Order-of-magnitude capital needs and use of funds for the next 24–36 months.
- Potential Founders / Talent → Profile of ideal founders / key early hires, plus any notable labs/teams that could spin this out.
- Sector/Vertical → 1–2 labeled sectors/subsectors (e.g. “Quantum simulation for drug discovery”, “Industrial photonics for semiconductor metrology”).
"""

    user_msg = f"""
Based on this scientific context:

{context}

Startup idea: "{idea}"
Problem (company-level): "{problem}"

Return each field exactly once, using *exactly* these field names:
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

Remember:
- Keep each answer on one line.
- Do not add extra fields, headings, bullets, or explanations.
"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": truncate_words(user_msg)},
        ],
        temperature=0.2,
    )
    output = resp.choices[0].message.content.strip()
    print("\n🎯 GPT Output Raw:\n", output)
    return output


# --- GPT logic: ONE-PAGER / DEEPTECH BRIEF (pitch-quality memo) ---
def generate_deeptech_brief(idea: str, problem: str, snippets: List[str]) -> str:
    context = "\n\n".join(f"- {s}" for s in snippets)
    prompt = f"""
You are drafting a concise **deeptech opportunity one-pager** for Quantonation about a potential startup.

Use the scientific context below (citations) + your own knowledge to write a **pitch-quality memo** that an investor could read as a standalone document.

Scientific context (snippets, may contain multiple papers or projects):
{context}

Startup idea: "{idea}"
Company-level problem statement: "{problem}"

Write the memo in **4 titled sections**, in this exact order:

1. Scientific Context & Problem
   - Briefly explain the scientific / technical background.
   - Connect it explicitly to the **company-level problem**: who suffers, why now, and why the problem is structurally hard.
   - When you reference papers from the context, cite them with a short bracket reference like [Smith et al., 2022 – MIT] and mention the team or lab.

2. Market Analysis
   - Describe the target customers, buying center, and initial wedge use case.
   - Quantify the opportunity with realistic TAM / SAM numbers and relevant geographies.
   - Explain *why now* (regulation, cost curves, enabling tech, geopolitical pressure, etc.).
   - Make this section specific enough that it reads like the “Market” slide of a pitch deck, not generic industry commentary.

3. Competitive Landscape & Positioning
   - List existing companies, consortia, and open-source projects that are relevant, with one line each on what they do.
   - Explain how this startup would position itself versus these players (complementary, orthogonal, or head-on).
   - Highlight 2–3 durable moats (IP, data, infra, partnerships, regulatory).

4. Key People & Ecosystem
   - Suggest relevant research groups, labs, and corporates to engage with (including those hinted at in the scientific context).
   - Mention profiles of ideal founders / early team (e.g. “ex-X lab + Y-industry operator”).
   - Add any key ecosystem nodes (accelerators, standardization bodies, early adopter corporates).

Global constraints:
- Total length ~600–900 words so it can be used as a one-pager.
- No bullet lists inside sections; use short paragraphs with clear topic sentences.
- Keep tone analytical, precise, and grounded (this is an internal VC memo, not marketing copy).
- Refer to the potential startup simply as “the company” or “this startup”, do not invent a new brand name.

Output only the memo, with the four section titles as level-2 headings (e.g. "## Scientific Context & Problem").
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": truncate_words(prompt)}],
        temperature=0.6,
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

    data = requests.post(
        f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query",
        headers=notion_headers,
    ).json()

    for item in data["results"]:
        page_id = item["id"]
        props = item["properties"]

        try:
            idea = props["Company Idea"]["title"][0]["text"]["content"]
        except Exception:
            print("⛔ Skipping item with no title")
            continue

        try:
            problem = props["Problem Statement"]["rich_text"][0]["text"]["content"]
        except Exception:
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
