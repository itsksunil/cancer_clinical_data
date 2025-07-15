import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.4
MAX_RESULTS = 50

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# Load and preprocess data
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # Filter and preprocess Q&A entries
    filtered_data = []
    for d in full_data:
        if "prompt" in d and "completion" in d:
            # Preprocess text - lowercase, remove special chars, etc.
            processed_prompt = re.sub(r'[^\w\s]', '', d["prompt"].lower()).strip()
            processed_completion = re.sub(r'[^\w\s]', '', d["completion"].lower()).strip()
            
            # Create metadata for each entry
            entry = {
                "original_prompt": d["prompt"],
                "original_completion": d["completion"],
                "processed_prompt": processed_prompt,
                "processed_completion": processed_completion,
                "source": d.get("source", "unknown"),
                "category": d.get("category", "general")
            }
            filtered_data.append(entry)
    
    return filtered_data

# Compute embeddings from processed prompts
@st.cache_data
def compute_embeddings(data):
    model = load_model()
    prompts = [d["processed_prompt"] for d in data]
    return model.encode(prompts, show_progress_bar=False)

# Streamlit UI
st.set_page_config(
    page_title="🧬 Advanced Cancer Q&A Semantic Search", 
    layout="centered",
    menu_items={
        'Get Help': 'https://www.cancer.gov/about-cancer/treatment/clinical-trials',
        'About': "This app provides semantic search capabilities for cancer clinical trial Q&A data."
    }
)

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
    
    # Category filter
    categories = st.multiselect(
        "Filter by category",
        options=["general", "treatment", "diagnosis", "symptoms", "prognosis"],
        default=["general", "treatment", "diagnosis", "symptoms", "prognosis"],
        help="Select categories to include in search"
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

# Main content
st.title("🧬 Advanced Cancer Clinical Q&A Search")
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
    AI-powered semantic search for cancer clinical trial questions and answers.<br>
    Search by meaning rather than just keywords.
    </div>
""", unsafe_allow_html=True)

# Load data and model
data = load_data()
prompt_embeddings = compute_embeddings(data)
model = load_model()

# Search functionality
query = st.text_input(
    "Ask your clinical question:", 
    placeholder="e.g. What are the side effects of chemotherapy?",
    help="Enter your question about cancer clinical trials"
)

if query:
    # Preprocess query
    processed_query = re.sub(r'[^\w\s]', '', query.lower()).strip()
    query_embedding = model.encode([processed_query])
    similarity_scores = cosine_similarity(query_embedding, prompt_embeddings)[0]

    # Sort results by similarity and apply filters
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    # Filter results by threshold and category
    filtered_results = []
    for idx in ranked_indices:
        if (similarity_scores[idx] >= threshold and 
            data[idx]["category"] in categories):
            filtered_results.append((idx, similarity_scores[idx]))
            if len(filtered_results) >= max_results:
                break
    
    # Display results
    if filtered_results:
        st.success(f"Found {len(filtered_results)} relevant results")
        
        # Group similar results
        grouped_results = defaultdict(list)
        for idx, score in filtered_results:
            key = data[idx]["processed_prompt"][:50]  # Group by prompt prefix
            grouped_results[key].append((idx, score))
        
        # Display grouped results
        for group in grouped_results.values():
            primary_idx, primary_score = group[0]
            entry = data[primary_idx]
            
            with st.expander(f"🔍 {entry['original_prompt']}", expanded=True):
                if show_scores:
                    st.caption(f"Similarity score: {primary_score:.2f}")
                
                # Highlight query terms if enabled
                completion_text = entry['original_completion']
                if highlight_terms:
                    for term in processed_query.split():
                        completion_text = re.sub(
                            f'({term})', 
                            r'<span style="background-color: #FFFF00">\1</span>', 
                            completion_text, 
                            flags=re.IGNORECASE
                        )
                
                st.markdown(f"**Answer:** {completion_text}", unsafe_allow_html=True)
                st.caption(f"Category: {entry['category'].title()} | Source: {entry['source']}")
                st.markdown("---")
    else:
        st.warning("No results found matching your criteria. Try:")
        st.markdown("- Lowering the similarity threshold")
        st.markdown("- Expanding the category filters")
        st.markdown("- Rewording your question")
else:
    st.info("💡 Tip: Ask questions in natural language, like:")
    st.markdown("- 'What are the latest treatments for breast cancer?'")
    st.markdown("- 'How is immunotherapy administered?'")
    st.markdown("- 'What side effects might I expect from chemotherapy?'")

# Add footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #666; font-size: 0.9em;'>
    Cancer Clinical Q&A Search • Powered by Sentence Transformers • Data from clinical trials
    </div>
""", unsafe_allow_html=True)
