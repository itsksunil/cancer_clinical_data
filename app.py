import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load only valid Q&A entries
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # Filter only Q&A format
    filtered_data = [d for d in full_data if "prompt" in d and "completion" in d]
    prompts = [d["prompt"] for d in filtered_data]
    return filtered_data, prompts

# Compute embeddings from prompts
@st.cache_data
def compute_embeddings(prompts):
    model = load_model()
    return model.encode(prompts, show_progress_bar=False)

# Streamlit UI
st.set_page_config(page_title="🧬 Cancer Q&A Semantic Search", layout="centered")
st.title(" Semantic Search on Cancer Clinical Data")
st.markdown(" Keyword based search from clinical trial,using AI-powered semantic search.")

# Load
data, prompts = load_data()
prompt_embeddings = compute_embeddings(prompts)

# Input
query = st.text_input("🔍 Ask your clinical question:")

if query:
    model = load_model()
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, prompt_embeddings)[0]

    # Sort all by similarity
    ranked_indices = np.argsort(similarity_scores)[::-1]

    # Slider for cutoff
    threshold = st.slider("🔎 Similarity threshold", 0.0, 1.0, 0.4, step=0.01)

    results_found = False
    for rank, idx in enumerate(ranked_indices, 1):
        score = similarity_scores[idx]
        if score >= threshold:
            results_found = True
            st.success(f"🔹 Rank {rank} | Similarity: {score:.2f}")
            st.markdown(f"**Prompt:** {data[idx]['prompt']}")
            st.markdown(f"**Answer:** {data[idx]['completion']}")
            st.markdown("---")

    if not results_found:
        st.warning("❗ No results found above the threshold. Try lowering it.")
else:
    st.info("Enter a question above to search the dataset.")
