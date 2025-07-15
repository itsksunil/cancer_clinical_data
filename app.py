import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load the dataset
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [entry["prompt"] for entry in data if "prompt" in entry]
    return data, prompts

# Compute embeddings (only for prompts)
@st.cache_data
def compute_embeddings(prompts):
    model = load_model()
    return model.encode(prompts, show_progress_bar=False)

# UI setup
st.set_page_config(page_title="🧬 Cancer Clinical Q&A", layout="centered")
st.title("🧠 Semantic Search: Cancer Clinical Trial Q&A")
st.markdown("Type any cancer-related clinical question. This app uses semantic embeddings to return the most relevant prompts and answers.")

# Load data and embeddings
data, prompts = load_data()
prompt_embeddings = compute_embeddings(prompts)

# Get user query
query = st.text_input("🔍 Ask your clinical question:")

if query:
    model = load_model()
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]

    # Sort all matches by similarity (descending)
    ranked_indices = np.argsort(similarities)[::-1]

    # Threshold slider
    threshold = st.slider("📊 Similarity threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

    found = False
    for rank, idx in enumerate(ranked_indices, 1):
        similarity_score = similarities[idx]
        if similarity_score >= threshold:
            found = True
            st.success(f"🔹 Rank {rank} (Score: {similarity_score:.2f})")
            st.markdown(f"**Prompt:** {data[idx]['prompt']}")
            st.markdown(f"**Answer:** {data[idx]['completion']}")
            st.markdown("---")
    if not found:
        st.warning("No results found above the threshold. Try lowering it.")
else:
    st.info("Enter a question to view similar results.")
