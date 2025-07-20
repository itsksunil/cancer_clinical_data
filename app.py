import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os
import requests
from urllib.parse import urlparse

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_home' not in st.session_state:
    st.session_state.show_home = True
if 'min_score' not in st.session_state:
    st.session_state.min_score = 1
if 'keyword_filters' not in st.session_state:
    st.session_state.keyword_filters = []
if 'cancer_type_filter' not in st.session_state:
    st.session_state.cancer_type_filter = []
if 'gene_filter' not in st.session_state:
    st.session_state.gene_filter = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'cancer_types' not in st.session_state:
    st.session_state.cancer_types = []
if 'genes' not in st.session_state:
    st.session_state.genes = []
if 'remote_data_url' not in st.session_state:
    st.session_state.remote_data_url = ""
if 'use_remote_data' not in st.session_state:
    st.session_state.use_remote_data = False

# Constants - now using DATA_FILES list
DATA_FILES = ["cancer_clinical_dataset.json", "NCT02394626.json"]  # Add all your JSON files here
HISTORY_FILE = "search_history.json"
REMOTE_DATA_TIMEOUT = 10

def load_search_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_search_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f)
    except:
        pass

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def fetch_remote_data(url):
    if not is_valid_url(url):
        st.error("Invalid URL provided")
        return None
    
    try:
        response = requests.get(url, timeout=REMOTE_DATA_TIMEOUT)
        response.raise_for_status()
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            st.error("Remote data is not in JSON format")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching remote data: {str(e)}")
        return None

# Modified to load multiple files
def load_data_source():
    if st.session_state.use_remote_data and st.session_state.remote_data_url:
        return fetch_remote_data(st.session_state.remote_data_url)
    else:
        combined_data = []
        for data_file in DATA_FILES:
            if os.path.exists(data_file):
                try:
                    with open(data_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            combined_data.extend(file_data)
                        else:
                            st.warning(f"File {data_file} doesn't contain a JSON array")
                except Exception as e:
                    st.error(f"Error loading {data_file}: {str(e)}")
            else:
                st.warning(f"Data file not found: {data_file}")
        
        return combined_data if combined_data else None

@st.cache_data
def load_and_index_data():
    try:
        raw_data = load_data_source()
        if raw_data is None:
            return None, None, [], [], []
        
        clean_data = []
        word_index = defaultdict(list)
        cancer_types = set()
        genes = set()
        all_prompts = []

        for idx, entry in enumerate(raw_data):
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                
                entry_cancer_types = []
                if "cancer_type" in entry:
                    entry_cancer_types = [ct.strip() for ct in str(entry["cancer_type"]).split(",")]
                    cancer_types.update(entry_cancer_types)
                
                entry_genes = []
                if "genes" in entry:
                    entry_genes = [g.strip() for g in str(entry["genes"]).split(",")]
                    genes.update(entry_genes)
                
                clean_data.append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "cancer_type": ", ".join(entry_cancer_types) if entry_cancer_types else "",
                    "genes": ", ".join(entry_genes) if entry_genes else ""
                })
                all_prompts.append(entry["prompt"])

                for word in set(entry["prompt"].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)
                for word in set(entry["completion"].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None, [], [], []
        
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, [], [], []

# Rest of your functions remain the same (keyword_search, filter_results, show_settings, show_home, show_results, etc.)
# ...

def main():
    st.set_page_config(
        page_title="🧬 Cancer Clinical Trial Search",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False

    if not st.session_state.search_history:
        st.session_state.search_history = load_search_history()

    if not st.session_state.suggestions or not st.session_state.cancer_types or not st.session_state.genes:
        data, word_index, cancer_types, genes, suggestions = load_and_index_data()
        if suggestions:
            st.session_state.suggestions = suggestions
        if cancer_types:
            st.session_state.cancer_types = cancer_types
        if genes:
            st.session_state.genes = genes

    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Cancer+Search", width=150)
        st.title("Navigation")

        if st.session_state.show_settings:
            if st.button("🏠 Home"):
                st.session_state.show_settings = False
                st.session_state.show_home = True
                st.rerun()
        elif not st.session_state.show_home:
            if st.button("🏠 Home"):
                st.session_state.show_home = True
                st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This platform helps researchers:
        - Find clinical trial details
        - Access treatment outcomes
        - Download structured data
        - Filter by cancer types and genes
        """)

    if st.session_state.show_settings:
        show_settings()
    elif st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
