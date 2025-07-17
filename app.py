import streamlit as st
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# --- Load Embedding Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Load and Clean JSON Dataset ---
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [d for d in raw_data if isinstance(d, dict) and "prompt" in d and "completion" in d]

# --- Compute Embeddings for Prompts ---
@st.cache_data
def compute_prompt_embeddings(prompts):
    model = load_model()
    return model.encode(prompts, show_progress_bar=False)

# --- Query Logger ---
def log_query(query, top_prompt):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "top_prompt": top_prompt
    }
    with open("query_log.csv", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_data) + "\n")

# --- Streamlit UI ---
st.set_page_config(page_title="🧬 Neural Search for Cancer Trials", layout="wide")
st.title("🔍 LLM-Enhanced Cancer Clinical Q&A Explorer")
st.markdown("Search structured clinical answers using deep neural embeddings (no fuzzy match, no GPT required).")

# Load and prep
data = load_data()
prompts = [item["prompt"] for item in data]
model = load_model()
prompt_embeddings = compute_prompt_embeddings(prompts)

# User Query
query = st.text_input("Ask your clinical question (e.g. 'What is OS in IMpower010?'):")

if query:
    # Embed and search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    top_results = [data[i] for i in ranked_indices[:5]]

    st.markdown("### 🔎 Top 5 Matches")
    for rank, idx in enumerate(ranked_indices[:5], start=1):
        score = similarities[idx]
        entry = data[idx]
        st.success(f"🔹 Rank {rank} | Similarity Score: {score:.2f}")
        st.markdown(f"**Prompt:** {entry['prompt']}")
        st.markdown(f"**Answer:** {entry['completion']}")
        with st.expander("📄 JSON View"):
            st.json(entry)
        st.markdown("---")

    # Log first result
    log_query(query, top_results[0]["prompt"])

    # Downloads
    df = pd.DataFrame(top_results)
    json_str = json.dumps(top_results, indent=2)
    csv_str = df.to_csv(index=False)

    st.markdown("### 📁 Download Your Results")
    st.download_button("⬇️ Download JSON", data=json_str, file_name="top_results.json", mime="application/json")
    st.download_button("⬇️ Download CSV", data=csv_str, file_name="top_results.csv", mime="text/csv")
else:
    st.info("Enter a clinical question to begin semantic search.")
