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

Requirements:
- 2–4 sentences, no bullets, no headings.
- Explicitly state **who** has the problem (customer segment, typical role, industry).
- Describe the current workflow / status quo and why it is inadequate.
- Quantify the pain with order-of-magnitude metrics (time, cost, yield loss, error rates, risk, etc.).
- Anchor it in real-world constraints (scientific, engineering, regulatory, or infrastructure limitations).
- Make the problem urgent and recurring, not a nice-to-have.
- Tone: clear, analytical, investor-grade. No marketing fluff.

Output only the final paragraph. No introductions, no labels, no closing.
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
You are a senior deeptech VC partner at Quantonation, writing **internal memo fields** for a potential investment.

Goal:
Produce **concise, pitch-quality answers** that can be copied as-is into an internal one-pager. 
Each field should be concrete, specific to THIS idea, and non-generic.

Style rules:
- Each field value is a **single line** (no line breaks).
- 1–3 short sentences per field.
- Prefer numbers, examples, named segments and use-cases over buzzwords.
- If you must estimate, use realistic order-of-magnitude ranges and clearly mark them as approximate.
- NEVER answer with "not specified", "TBD", "unknown" or similar. If data is missing, infer a plausible range and say it is approximate.
- Do not greet, do not sign, do not add headings or commentary.

You must respond **only** in this exact format, one field per line:
Field: Value
"""

    user_msg = f"""
Scientific context (snippets, may contain multiple papers or projects):

{context}

Startup idea: "{idea}"
Company-level problem statement: "{problem}"

Using the context above and your own knowledge, fill in ALL of the following fields.
Use **exactly** these field names, with exactly this spelling and punctuation, and in exactly this order:

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

Content expectations for each field:
- Technology Leveraged → Concrete description of the core scientific/engineering approach and why it is suited to the problem.
- Market Size → Approximate TAM and initial SAM (with currency and geographies) plus 1 sentence on the adoption driver.
- Competitive Advantage → 2–3 short differentiators separated by semicolons, focused on what is hard to copy.
- Feasibility Score (1–10) → "X/10 –" followed by a brief justification referencing technical and execution risk.
- Investment Thesis Fit → 1–2 sentences on why this fits a deeptech / Quantonation-type thesis.
- Next Steps → 2–4 concrete milestones separated by semicolons.
- Problem Severity (1–10) → "X/10 –" with a brief justification referencing economic or strategic pain.
- Tech Readiness Level (TRL 1–9) → "X/9 –" with a short justification referencing prototype/lab/pilot status.
- Strategic Partner Ideas → Names or types of specific potential partners and what they would bring.
- Funding Needs → Order-of-magnitude capital and main use of funds over the next 24–36 months.
- Potential Founders / Talent → Profiles of ideal founders / early hires and any notable labs/teams that could spin this out.
- Sector/Vertical → 1–2 labeled sectors/subsectors (e.g. "Quantum sensing for inertial navigation", "Aerospace & defense").

Return:
- Exactly 12 lines, one per field.
- No extra text before or after.
"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": truncate_words(user_msg)},
        ],
        temperature=0.25,
    )
    output = resp.choices[0].message.content.strip()
    print("\n🎯 GPT Output Raw:\n", output)
    return output


# --- GPT logic: ONE-PAGER / DEEPTECH BRIEF (pitch-quality memo) ---
def generate_deeptech_brief(idea: str, problem: str, snippets: List[str]) -> str:
    context = "\n\n".join(f"- {s}" for s in snippets)
    prompt = f"""
You are drafting a concise **deeptech investment memo** for Quantonation about a potential startup.

This will be saved as a Notion subpage and should read like a **standalone one-pager memo**, not like an email.
Therefore:
- Do NOT greet anyone (no "Hi", no "Dear").
- Do NOT address a specific recipient.
- Do NOT sign off (no "Best regards", no name at the end).
- Just write the memo itself: headings + paragraphs.

Scientific context (snippets, may contain multiple papers or projects):
{context}

Startup idea: "{idea}"
Company-level problem statement: "{problem}"

Write the memo with **4 sections**, using Markdown level-2 headings (##):

## Scientific Context & Problem
Explain the relevant scientific and technical background in clear language. 
Connect the background explicitly to the company-level problem: who is affected, how current solutions fail, and why the problem is structurally hard (physics, engineering, regulation, infrastructure, etc.).
When referencing items from the scientific context, cite them concisely like [Smith et al., 2022 – MIT] and mention the team or lab.
Avoid over-focusing on a single paper when others are relevant.

## Market Analysis
Describe the target customers and buying centers, the initial wedge use case, and the broader expansion path.
Quantify the opportunity (TAM / SAM and logical order-of-magnitude figures) with at least one geography.
Explain **why now** (technological maturity, regulation, cost curves, supply-chain or geopolitical drivers, etc.).
This section should read like the “Market” and “Why now” slides of a pitch deck, not generic industry commentary.

## Competitive Landscape & Positioning
List the key existing companies, consortia, and open-source projects in this space, each with one short description line.
Explain how this potential startup would position itself relative to them (complementary, orthogonal, or head-on competition).
Highlight 2–3 durable moats (IP, data, infrastructure, regulatory position, ecosystem) and how they could realistically be built.

## Key People & Ecosystem
Suggest relevant research groups, labs, and corporates to engage with (including those implied by the scientific context).
Describe the ideal founder and early-team profiles (e.g. "postdoc from X lab + ex-Y industry operator").
Mention key ecosystem nodes (accelerators, standardization bodies, early adopter corporates) that would materially de-risk the opportunity.

Global constraints:
- Total length ~700–1,000 words.
- Use short paragraphs; no bullet lists.
- Tone: analytical, neutral, and precise – this is an internal investment memo, not marketing copy.
- Refer to the potential startup as “the company” or “this startup”; do not invent a brand name.

Output only the memo, starting with "## Scientific Context & Problem".
No preamble, no meta-comments, no closing signature.
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
