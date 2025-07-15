import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load the JSON data only once
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [entry["prompt"] for entry in data if "prompt" in entry]
    return data, prompts

# Compute prompt embeddings (only pass hashable 'prompts')
@st.cache_data
def compute_embeddings(prompts):
    model = load_model()
    return model.encode(prompts, show_progress_bar=False)

# App UI
st.set_page_config(page_title="🧬 Cancer Clinical Q&A", layout="centered")
st.title("🧠 Semantic Search: Cancer Clinical Trial Q&A")
st.markdown("Ask a clinical question and discover the most relevant answers using AI-based semantic search.")

# Load data
data, prompts = load_data()
embeddings = compute_embeddings(prompts)

# Get user input
query = st.text_input("🔍 Type your clinical question here:")

if query:
    model = load_model()
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-3:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        st.success(f"✅ Match {rank}")
        st.markdown(f"**Prompt:** {data[idx]['prompt']}")
        st.markdown(f"**Answer:** {data[idx]['completion']}")
        st.markdown("---")
else:
    st.info("Enter a clinical question to get started.")
