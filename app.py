import streamlit as st
from PIL import Image
import datetime
import requests
import os
import json

from quantonation_agent import (
    generate_problem_statement,
    generate_gpt_output,
    generate_deeptech_brief,
    update_problem_statement,
    update_notion_properties,
    create_notion_subpage,
    search_corpus,
    build_faiss_index,
    load_faiss_index,
    parse_gpt_response,
    notion_headers,
    NOTION_DATABASE_ID
)

# ----- Page Config -----
st.set_page_config(page_title="Quantonation Virtual Studio", layout="wide")

# ----- Logo -----
logo = Image.open("dyybccql1d15pxbo0ecu-3.png")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(logo, width=250)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("## 🧠 Quantonation Virtual Studio")

# ----- Load FAISS Index -----
if os.path.exists("faiss.index") and os.path.exists("corpus_texts.json"):
    index, corpus_texts = load_faiss_index()
    st.success("FAISS index loaded.")
else:
    index, corpus_texts = build_faiss_index()
    st.success("FAISS index built.")

# ----- Define the Sidebar Function -----
def show_faiss_sidebar(index, avg_chunks_per_paper=20):
    """
    Display FAISS index size and estimated paper count in Streamlit sidebar.

    Parameters:
    - index: a loaded FAISS index object
    - avg_chunks_per_paper: estimated average number of chunks per paper
    """
    vector_count = index.ntotal
    estimated_papers = vector_count // avg_chunks_per_paper

    with st.sidebar:
        st.markdown("## 📊 FAISS Monitor")
        st.metric(label="Vectors Indexed", value=vector_count)
        st.metric(label="Estimated Papers", value=estimated_papers)
        st.caption(f"📝 Estimation based on ~{avg_chunks_per_paper} chunks per paper.")

# ----- Load FAISS Index -----
if os.path.exists("faiss.index") and os.path.exists("corpus_texts.json"):
    index, corpus_texts = load_faiss_index()
    st.success("FAISS index loaded.")
else:
    index, corpus_texts = build_faiss_index()
    st.success("FAISS index built.")

# ✅ Call the sidebar function
show_faiss_sidebar(index)



# ----- Fetch Existing Notion Pages -----
existing_pages = []
query_payload = {"page_size": 100}
response = requests.post(
    f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query",
    headers=notion_headers,
    json=query_payload
)
if response.status_code == 200:
    for page in response.json().get("results", []):
        title_prop = page["properties"].get("Company Idea", {}).get("title", [])
        if title_prop:
            title_text = title_prop[0]["text"]["content"]
            existing_pages.append((title_text, page["id"]))
else:
    st.error("Failed to fetch existing Notion pages")

# ----- Update Existing Page -----
st.subheader("🔁 Update Existing Page")
selected_title = st.selectbox("Choose an existing idea to update", [title for title, _ in existing_pages])
existing_id = next((pid for title, pid in existing_pages if title == selected_title), None)
update_now = st.checkbox("Enrich this existing page now", value=False)

if update_now and existing_id.strip():
    try:
        page_data = requests.get(f"https://api.notion.com/v1/pages/{existing_id.strip()}", headers=notion_headers)
        if page_data.status_code == 200:
            page = page_data.json()
            idea = page["properties"]["Company Idea"]["title"][0]["text"]["content"]
            with st.spinner(f"Updating existing idea: {idea}"):
                problem = generate_problem_statement(idea)
                update_problem_statement(existing_id, problem)

                context_snippets = search_corpus(index, corpus_texts, f"{idea}. {problem}", top_k=5)
                gpt_resp = generate_gpt_output(idea, problem, context_snippets)
                updates = parse_gpt_response(gpt_resp)
                update_notion_properties(existing_id, updates)

                memo = generate_deeptech_brief(idea, problem, context_snippets)
                title = f"{datetime.datetime.now().strftime('%Y-%m-%d')} – Memo: {idea[:60]}"
                create_notion_subpage(existing_id, title, memo)
                st.success("✅ Existing Notion page updated.")
        else:
            st.error(f"Failed to fetch page: {page_data.text}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ----- Add New Idea -----
st.subheader("➕ Add New Startup Idea")
with st.form("new_theme_form"):
    new_idea = st.text_input("Enter new startup idea (Company Idea title)")
    enrich_now = st.checkbox("Immediately enrich this idea after adding", value=True)
    submitted = st.form_submit_button("Add Idea")

if submitted:
    if new_idea.strip():
        new_page_payload = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "Company Idea": {"title": [{"text": {"content": new_idea.strip()}}]}
            }
        }
        response = requests.post("https://api.notion.com/v1/pages", headers=notion_headers, json=new_page_payload)
        if response.status_code == 200:
            new_page = response.json()
            new_id = new_page["id"]
            st.success(f"✅ Created new page for: {new_idea}")

            if enrich_now:
                with st.spinner("Enriching via GPT..."):
                    problem = generate_problem_statement(new_idea)
                    update_problem_statement(new_id, problem)

                    context_snippets = search_corpus(index, corpus_texts, f"{new_idea}. {problem}", top_k=5)
                    gpt_resp = generate_gpt_output(new_idea, problem, context_snippets)
                    updates = parse_gpt_response(gpt_resp)
                    update_notion_properties(new_id, updates)

                    memo = generate_deeptech_brief(new_idea, problem, context_snippets)
                    title = f"{datetime.datetime.now().strftime('%Y-%m-%d')} – Memo: {new_idea[:60]}"
                    create_notion_subpage(new_id, title, memo)

                    st.success("✅ Notion page enriched successfully.")
        else:
            st.error(f"❌ Failed to create new theme: {response.text}")
    else:
        st.warning("⚠️ Please enter a valid idea title.")

# ----- Upload PDF Papers -----
st.markdown("### 📄 Upload Academic Papers (PDF)")

uploaded_files = st.file_uploader(
    "Upload PDF papers to add to GPT context",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Parsing and indexing uploaded papers..."):
        from quantonation_agent import inject_pdf_into_faiss
        index, corpus_texts = inject_pdf_into_faiss(uploaded_files, index, corpus_texts)
    st.success("✅ PDFs parsed and added to context.")

