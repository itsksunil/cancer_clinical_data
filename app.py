import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load the JSON dataset
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [entry["prompt"] for entry in data if "prompt" in entry]
    return data, prompts

# Compute prompt embeddings
@st.cache_data
def compute_embeddings(prompts, model):
    return model.encode(prompts, show_progress_bar=False)

# Streamlit Page Settings
st.set_page_config(page_title="🧬 Cancer Clinical Q&A - Semantic Search", layout="centered")
st.title("🧠 Cancer Clinical Trial Semantic Q&A")
st.markdown("Ask any question about cancer trials, biomarkers, immune responses, or clinical outcomes.")

# Load model and data
model = load_model()
data, prompts = load_data()
prompt_embeddings = compute_embeddings(prompts, model)

# User query input
query = st.text_input("🔍 Ask your clinical question here:")

if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]  # Top 3 most similar prompts

    for i, idx in enumerate(top_indices, 1):
        st.success(f"✅ Match {i}")
        st.markdown(f"**Prompt:** {data[idx]['prompt']}")
        st.markdown(f"**Answer:** {data[idx]['completion']}")
        st.markdown("---")
else:
    st.info("Please enter a clinical question to see relevant answers.")
