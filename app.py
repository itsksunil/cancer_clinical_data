import streamlit as st
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Embedding Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

# --- Load and Filter Dataset ---
@st.cache_data
def load_data():
    try:
        with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return [d for d in raw_data if isinstance(d, dict) and "prompt" in d and "completion" in d]
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return []

# --- Compute Embeddings ---
@st.cache_data
def compute_prompt_embeddings(prompts):
    model = load_model()
    return model.encode(prompts, show_progress_bar=False)

# --- App UI ---
st.set_page_config(page_title="🧠 Cancer LLM Neural Search", layout="wide")
st.title("🧬 Atezolizumab Clinical Q&A Assistant (LLM-Powered)")
st.markdown("Ask your clinical question about Atezolizumab trials, immune mechanisms, PD-L1, or outcomes.")

# Load data and embeddings
data = load_data()
prompts = [item["prompt"] for item in data]
prompt_embeddings = compute_prompt_embeddings(prompts)
model = load_model()

# User query
query = st.text_input("🔍 Ask your clinical question:")

if query:
    # Embed query and calculate similarities
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    top_results = [data[idx] for idx in ranked_indices[:5]]

    # Display Top Results
    st.markdown("### 🔎 Top 5 Relevant Results")
    for rank, idx in enumerate(ranked_indices[:5], start=1):
        score = similarities[idx]
        entry = data[idx]
        st.success(f"🔹 Rank {rank} | Similarity: {score:.2f}")
        st.markdown(f"**🧾 Prompt:** {entry['prompt']}")
        st.markdown(f"**💬 Answer:** {entry['completion']}")
        with st.expander("📄 Raw JSON"):
            st.json(entry)
        st.markdown("---")

    # Prepare download formats
    df = pd.DataFrame(top_results)
    json_str = json.dumps(top_results, indent=2)
    csv_str = df.to_csv(index=False)

    # Download buttons
    st.markdown("### 📁 Download Results")
    st.download_button("⬇️ Download JSON", data=json_str, file_name="top_results.json", mime="application/json")
    st.download_button("⬇️ Download CSV", data=csv_str, file_name="top_results.csv", mime="text/csv")
else:
    st.info("Enter a free-form clinical question to begin neural search.")
