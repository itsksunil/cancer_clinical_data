import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'min_score' not in st.session_state:
    st.session_state.min_score = 0.3

# Constants
DATA_FILE = "cancer_clinical_dataset.json"
HISTORY_FILE = "search_history.json"

# Load data
@st.cache_data
def load_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict) and "prompt" in d and "completion" in d]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []

# Simple keyword search
def search(query, dataset, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    results = []
    for idx, score in enumerate(similarities):
        if score > st.session_state.min_score:
            results.append({
                "entry": dataset[idx],
                "score": score
            })
    return sorted(results, key=lambda x: x["score"], reverse=True)

# Main app
def main():
    st.set_page_config(
        page_title="Cancer Clinical Search",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Cancer Clinical Search")
    st.write("Search through clinical cancer Q&A data")
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Prepare TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([d["prompt"] + " " + d["completion"] for d in data])
    
    # Search input
    query = st.text_input("Search clinical questions:", 
                         placeholder="e.g., breast cancer treatment")
    
    # Search button
    if st.button("Search") and query:
        st.session_state.current_query = query
        st.session_state.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.spinner("Searching..."):
            results = search(query, data, vectorizer, tfidf_matrix)
            
            if results:
                st.success(f"Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {result['score']:.2f})"):
                        st.write(f"**Question:** {result['entry']['prompt']}")
                        st.write(f"**Answer:** {result['entry']['completion']}")
            else:
                st.warning("No results found. Try a different query.")
    
    # Search history
    if st.session_state.search_history:
        with st.expander("Search History"):
            for i, search in enumerate(reversed(st.session_state.search_history), 1):
                if st.button(f"{i}. {search['query']}", key=f"history_{i}"):
                    st.session_state.current_query = search["query"]

if __name__ == "__main__":
    main()
