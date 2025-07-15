import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load the dataset once
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [entry["prompt"] for entry in data]
    return data, prompts

# Precompute embeddings
@st.cache_data
def compute_embeddings(prompts, model):
    return model.encode(prompts)

# Initialize
st.set_page_config(page_title="Cancer Clinical Q&A - Semantic Search", layout="centered")
st.title("🧬 Semantic Search in Cancer Clinical Q&A")
st.markdown("Ask your clinical question and find the most relevant answers using AI-based search.")

# Load data and model
data, prompts = load_data()
model = load_model()
prompt_embeddings = compute_embeddings(prompts, model)

# Input box
query = st.text_input("🔍 Ask a clinical question:")

if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]
    top_k = similarities.argsort()[-3:][::-1]  # Top 3 results

    for idx in top_k:
        st.success("✅ Match Found")
        st.markdown(f"**Prompt:** {data[idx]['prompt']}")
        st.markdown(f"**Answer:** {data[idx]['completion']}")
        st.markdown("---")
else:
    st.info("Enter a question to see results.")
