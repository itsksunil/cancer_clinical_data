import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
from datetime import datetime

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.4
MAX_RESULTS = 50

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# Load and preprocess data with metadata extraction
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)

    filtered_data = []
    for d in full_data:
        if "prompt" in d and "completion" in d:
            # Extract potential metadata from prompt/completion
            metadata = {
                "source": d.get("source", "unknown"),
                "category": d.get("category", "general"),
                "gene": d.get("gene", ""),
                "tumors": d.get("tumors", []),
                "mutations": d.get("mutations", []),
                "topic": d.get("topic", ""),
                "context": d.get("context", "")
            }
            
            # Preprocess text
            processed_prompt = re.sub(r'[^\w\s]', '', d["prompt"].lower()).strip()
            processed_completion = re.sub(r'[^\w\s]', '', d["completion"].lower()).strip()
            
            entry = {
                "original_prompt": d["prompt"],
                "original_completion": d["completion"],
                "processed_prompt": processed_prompt,
                "processed_completion": processed_completion,
                "metadata": metadata
            }
            filtered_data.append(entry)
    
    return filtered_data

# Compute embeddings
@st.cache_data
def compute_embeddings(data):
    model = load_model()
    prompts = [d["processed_prompt"] + " " + d["processed_completion"] for d in data]
    return model.encode(prompts, show_progress_bar=False)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stSlider>div>div>div>div {
        background: #4CAF50;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #ffffff;
    }
    .st-ae {
        background-color: #f0f2f6;
    }
    .stTitle>div>div>div {
        color: #2c3e50;
    }
    .css-1aumxhk {
        background-color: #2c3e50;
        color: white;
    }
    .highlight {
        background-color: #FFF59D;
        padding: 0.1em 0.2em;
        border-radius: 3px;
    }
    .metadata-chip {
        display: inline-block;
        background-color: #e0f7fa;
        padding: 0.2em 0.5em;
        margin: 0.2em;
        border-radius: 15px;
        font-size: 0.8em;
        color: #00796b;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for filters
with st.sidebar:
    st.title("🔍 Search Filters")
    threshold = st.slider(
        "Similarity threshold", 
        0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
        help="Higher values return more precise but fewer results"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        max_results = st.number_input(
            "Maximum results to show",
            min_value=1,
            max_value=100,
            value=MAX_RESULTS,
            help="Limit the number of results displayed"
        )
        show_scores = st.checkbox(
            "Show similarity scores", 
            value=True,
            help="Display the cosine similarity score for each result"
        )
        highlight_terms = st.checkbox(
            "Highlight matching terms",
            value=True,
            help="Highlight query terms in the results"
        )
        show_metadata = st.checkbox(
            "Show metadata",
            value=True,
            help="Display gene, tumor type, and other metadata when available"
        )
        search_mode = st.radio(
            "Search mode",
            options=["Semantic (recommended)", "Keyword"],
            index=0,
            help="Semantic search understands meaning, keyword search looks for exact terms"
        )

# Main content
st.title("🧬 Advanced Cancer Clinical Q&A Search")
st.markdown("AI-powered semantic search for cancer clinical trial questions and answers.")

# Load data and model
data = load_data()
prompt_embeddings = compute_embeddings(data)
model = load_model()

# Search functionality
query = st.text_input(
    "🔍 Ask your clinical question:", 
    placeholder="e.g. What are the side effects of chemotherapy?",
    help="Enter your question about cancer clinical trials"
)

if query:
    start_time = datetime.now()
    
    if search_mode == "Semantic (recommended)":
        # Semantic search with embeddings
        query_embedding = model.encode([query])
        similarity_scores = cosine_similarity(query_embedding, prompt_embeddings)[0]
        ranked_indices = np.argsort(similarity_scores)[::-1]
    else:
        # Keyword search
        query_terms = re.findall(r'\w+', query.lower())
        ranked_indices = []
        similarity_scores = []
        
        for idx, entry in enumerate(data):
            text = entry["processed_prompt"] + " " + entry["processed_completion"]
            matches = sum(1 for term in query_terms if term in text)
            if matches > 0:
                ranked_indices.append(idx)
                similarity_scores.append(matches / len(query_terms))
        
        # Sort by match percentage
        ranked_indices = [x for _, x in sorted(zip(similarity_scores, ranked_indices), reverse=True)]
        similarity_scores = sorted(similarity_scores, reverse=True)
    
    # Filter results by threshold and limit to max_results
    filtered_results = []
    for rank, idx in enumerate(ranked_indices[:max_results*3], 1):  # Look at more initially to account for metadata filtering
        score = similarity_scores[idx] if search_mode == "Semantic (recommended)" else similarity_scores[rank-1]
        if score >= threshold:
            filtered_results.append((idx, score))
            if len(filtered_results) >= max_results:
                break
    
    # Display results
    if filtered_results:
        search_time = (datetime.now() - start_time).total_seconds()
        st.success(f"Found {len(filtered_results)} relevant results in {search_time:.2f} seconds")
        
        for idx, score in filtered_results:
            entry = data[idx]
            
            with st.expander(f"🔹 {entry['original_prompt']}", expanded=False):
                if show_scores:
                    st.caption(f"Similarity score: {score:.2f}")
                
                # Highlight query terms if enabled
                completion_text = entry['original_completion']
                if highlight_terms:
                    query_terms = re.findall(r'\w+', query.lower())
                    for term in query_terms:
                        if len(term) > 3:  # Only highlight longer terms
                            completion_text = re.sub(
                                f'({term})', 
                                r'<span class="highlight">\1</span>', 
                                completion_text, 
                                flags=re.IGNORECASE
                            )
                
                st.markdown(f"**Answer:** {completion_text}", unsafe_allow_html=True)
                
                # Show metadata if available
                if show_metadata and entry['metadata']:
                    metadata_html = "<div style='margin-top: 10px;'>"
                    if entry['metadata'].get('gene'):
                        metadata_html += f"<span class='metadata-chip'>Gene: {entry['metadata']['gene']}</span>"
                    if entry['metadata'].get('tumors'):
                        tumors = ", ".join(entry['metadata']['tumors'])
                        metadata_html += f"<span class='metadata-chip'>Tumors: {tumors}</span>"
                    if entry['metadata'].get('mutations'):
                        mutations = ", ".join(entry['metadata']['mutations'])
                        metadata_html += f"<span class='metadata-chip'>Mutations: {mutations}</span>"
                    if entry['metadata'].get('category'):
                        metadata_html += f"<span class='metadata-chip'>Category: {entry['metadata']['category']}</span>"
                    if entry['metadata'].get('topic'):
                        metadata_html += f"<span class='metadata-chip'>Topic: {entry['metadata']['topic']}</span>"
                    metadata_html += "</div>"
                    st.markdown(metadata_html, unsafe_allow_html=True)
                
                st.markdown("---")
    else:
        st.warning("No results found above the threshold. Try:")
        st.markdown("- Lowering the similarity threshold")
        st.markdown("- Broadening your search terms")
        st.markdown("- Trying the keyword search mode")
else:
    st.info("💡 Tips for better searches:")
    st.markdown("- Ask questions in natural language, like 'What are the latest treatments for breast cancer?'")
    st.markdown("- Try both semantic and keyword search modes")
    st.markdown("- Use the filters to refine your results")

# Add footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #666; font-size: 0.9em;'>
    Cancer Clinical Q&A Search • Powered by Sentence Transformers • Data from clinical trials
    </div>
""", unsafe_allow_html=True)
