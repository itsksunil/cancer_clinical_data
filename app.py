# app.py

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from Levenshtein import distance as levenshtein_distance # For fuzzy matching

# For semantic search and visualization
import spacy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- NLTK Data Download (Run once if not already downloaded) ---
# This block ensures that necessary NLTK data (stopwords) are available.
# It will download them if they are not found, which might take a moment on first run.
try:
    _ = stopwords.words('english')
    _ = PorterStemmer()
except LookupError:
    import nltk
    st.warning("Downloading NLTK data (stopwords). This will only happen once.")
    nltk.download('stopwords')
    st.success("NLTK data downloaded. Please rerun the app if it doesn't automatically refresh.")

# --- spaCy Model Loading ---
# Load a pre-trained spaCy model for word embeddings.
# 'en_core_web_sm' is a small model, good for quick demos.
# For better semantic accuracy, consider 'en_core_web_md' or 'en_core_web_lg'
# You might need to install it: python -m spacy download en_core_web_sm
@st.cache_resource # Use st.cache_resource for models/heavy objects
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        st.stop() # Stop the app if model is not available
nlp = load_spacy_model()

# --- Session State Initialization ---
# Initialize all necessary variables in Streamlit's session state.
# Session state persists values across reruns of the app.
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_home' not in st.session_state:
    st.session_state.show_home = True
if 'min_keyword_score' not in st.session_state: # Renamed for clarity
    st.session_state.min_keyword_score = 1
if 'min_semantic_score' not in st.session_state: # New semantic score filter
    st.session_state.min_semantic_score = 0.0 # Range 0.0 to 1.0
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

# --- Constants ---
DATA_FILE = "cancer_clinical_dataset.json" # Name of your JSON dataset file
HISTORY_FILE = "search_history.json"      # File to store search history
LEVENSHTEIN_THRESHOLD = 2                 # Max Levenshtein distance for fuzzy matching (tune as needed)

# --- NLP Tools Initialization ---
stemmer = PorterStemmer()                  # Initialize the Porter Stemmer
stop_words = set(stopwords.words('english')) # Load English stop words into a set for fast lookup

# --- Utility Functions for Text Preprocessing ---

def preprocess_text(text):
    """
    Converts text to lowercase, removes punctuation, stems words, and removes stop words.
    This function is used for general text cleaning.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation (keep alphanumeric and spaces)
    words = [stemmer.stem(word) for word in text.split() if word.strip() and word not in stop_words]
    return " ".join(words)

def get_clean_words(text):
    """
    Returns a set of stemmed, non-stop words from a given text.
    Useful for creating word sets for comparison.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(stemmer.stem(word) for word in text.split() if word.strip() and word not in stop_words)

# --- Search History Management ---

def load_search_history():
    """Loads search history from a JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load search history: {e}")
            return []
    return []

def save_search_history():
    """Saves current search history to a JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f)
    except Exception as e:
        st.warning(f"Could not save search history: {e}")

# --- Data Loading and Indexing ---

@st.cache_data # Cache the data loading and indexing for performance
def load_and_index_data():
    """
    Loads the JSON dataset, cleans it, creates a word index for searching,
    extracts unique cancer types and genes, and generates spaCy document objects for semantic search.
    """
    try:
        if not os.path.exists(DATA_FILE):
            st.error(f"Data file '{DATA_FILE}' not found. Please ensure it's in the same directory.")
            return None, None, [], [], []

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        clean_data = []
        word_index = defaultdict(list) # Maps stemmed words to a list of document IDs
        cancer_types = set()
        genes = set()
        all_prompts = [] # Used for generating random suggestions

        for idx, entry in enumerate(raw_data):
            # Ensure entry is a dictionary and has 'prompt' and 'completion' fields
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                prompt = str(entry["prompt"]).strip()
                completion = str(entry["completion"]).strip()
                
                # Extract and clean metadata (cancer_type, genes)
                entry_cancer_types = [ct.strip() for ct in str(entry.get("cancer_type", "")).split(",") if ct.strip()]
                cancer_types.update(entry_cancer_types)
                
                entry_genes = [g.strip() for g in str(entry.get("genes", "")).split(",") if g.strip()]
                genes.update(entry_genes)
                
                # Store cleaned entry in our main data list
                clean_entry = {
                    "prompt": prompt,
                    "completion": completion,
                    "cancer_type": ", ".join(entry_cancer_types) if entry_cancer_types else "",
                    "genes": ", ".join(entry_genes) if entry_genes else ""
                }
                clean_data.append(clean_entry)
                all_prompts.append(prompt)

                # Index stemmed words from both prompt and completion for keyword search
                processed_prompt_words = get_clean_words(prompt)
                processed_completion_words = get_clean_words(completion)

                for word in processed_prompt_words.union(processed_completion_words):
                    if len(word) > 2: # Only index words longer than 2 characters
                        word_index[word].append(idx)

        # Generate spaCy document objects for semantic similarity calculation
        # Concatenate prompt and completion for a comprehensive document representation
        for entry in clean_data:
            entry["spacy_doc"] = nlp(entry["prompt"] + " " + entry["completion"])

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset. Please check your JSON file.")
            return None, None, [], [], []
        
        # Generate random suggestions from the prompts
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        # Return processed data, index, and metadata
        return clean_data, word_index, sorted(list(cancer_types)), sorted(list(genes)), random_suggestions
    
    except json.JSONDecodeError:
        st.error(f"Error: '{DATA_FILE}' is not a valid JSON file. Please check its format.")
        return None, None, [], [], []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {str(e)}")
        return None, None, [], [], []

# --- Search Logic (Combined Keyword and Semantic) ---

def combined_search(query, dataset, word_index, nlp_model):
    """
    Performs a combined keyword and semantic search on the dataset.
    Results are scored based on both exact/fuzzy keyword matches and semantic similarity.
    """
    if not query or not dataset:
        return []
    
    # Preprocess the query for keyword search
    query_words_stemmed = get_clean_words(query)
    
    # Create spaCy doc for the query for semantic search
    query_doc = nlp_model(query)

    # Dictionary to store match information for each document
    doc_results = defaultdict(lambda: {
        "keyword_score": 0,
        "semantic_score": 0.0,
        "total_score": 0.0,
        "matched_keywords": set(),
        "prompt_matches": 0,
        "completion_matches": 0
    })
    
    # Iterate through each document in the dataset
    for doc_id, entry in enumerate(dataset):
        # --- Keyword Matching ---
        processed_prompt_words = get_clean_words(entry["prompt"])
        processed_completion_words = get_clean_words(entry["completion"])
        
        current_matched_keywords = set()

        for q_word in query_words_stemmed:
            # 1. Exact Match (higher score)
            if q_word in processed_prompt_words:
                doc_results[doc_id]["prompt_matches"] += 1
                current_matched_keywords.add(q_word)
            if q_word in processed_completion_words:
                doc_results[doc_id]["completion_matches"] += 1
                current_matched_keywords.add(q_word)

            # 2. Fuzzy Match (Levenshtein distance - partial score)
            found_fuzzy_prompt = False
            for p_word in processed_prompt_words:
                if levenshtein_distance(q_word, p_word) <= LEVENSHTEIN_THRESHOLD:
                    doc_results[doc_id]["prompt_matches"] += 0.5
                    current_matched_keywords.add(p_word)
                    found_fuzzy_prompt = True
                    break

            if not found_fuzzy_prompt:
                for c_word in processed_completion_words:
                    if levenshtein_distance(q_word, c_word) <= LEVENSHTEIN_THRESHOLD:
                        doc_results[doc_id]["completion_matches"] += 0.25
                        current_matched_keywords.add(c_word)
                        break
        
        # Calculate keyword score with weights
        doc_results[doc_id]["keyword_score"] = (doc_results[doc_id]["prompt_matches"] * 2) + doc_results[doc_id]["completion_matches"]
        doc_results[doc_id]["matched_keywords"].update(current_matched_keywords)

        # --- Semantic Similarity ---
        # Calculate similarity between query and document's spacy_doc
        # Ensure both docs have vectors (some words might not have vectors in sm model)
        if query_doc.has_vector and entry["spacy_doc"].has_vector:
            semantic_similarity = query_doc.similarity(entry["spacy_doc"])
            doc_results[doc_id]["semantic_score"] = semantic_similarity
        else:
            doc_results[doc_id]["semantic_score"] = 0.0 # No vector, no semantic score

    ranked_results = []
    for doc_id, match_info in doc_results.items():
        # Combine keyword and semantic scores. You can adjust weights here.
        # For simplicity, let's use a sum, or a weighted sum.
        # Example: total_score = (keyword_weight * keyword_score) + (semantic_weight * semantic_score)
        # We need to normalize semantic score if it's not already in a comparable range.
        # SpaCy similarity is 0-1, so it's already normalized.
        
        # A simple additive model, giving keyword matches more direct impact
        # You can tune these weights (e.g., 10 * semantic_score to make it more impactful)
        match_info["total_score"] = match_info["keyword_score"] + (match_info["semantic_score"] * 5) 
        
        if match_info["total_score"] > 0: # Only include documents with a non-zero total score
            ranked_results.append({
                "entry": dataset[doc_id],
                "keyword_score": match_info["keyword_score"],
                "semantic_score": match_info["semantic_score"],
                "total_score": match_info["total_score"],
                "prompt_matches": match_info["prompt_matches"],
                "completion_matches": match_info["completion_matches"],
                "matched_keywords": match_info["matched_keywords"]
            })

    # Sort results by total score in descending order
    ranked_results.sort(key=lambda x: x["total_score"], reverse=True)
    return ranked_results

# --- Filtering Logic ---

def filter_results(results, min_keyword_score, min_semantic_score, keyword_filters, cancer_types, genes):
    """
    Filters a list of search results based on minimum keyword score, minimum semantic score,
    keyword filters, cancer type filters, and gene filters.
    """
    filtered = []
    
    # Preprocess filter keywords for consistent comparison (stemming + stop words)
    stemmed_keyword_filters = {stemmer.stem(kw.lower()) for kw in keyword_filters if kw.strip()}
    
    for result in results:
        entry = result["entry"]
        
        # 1. Apply Minimum Keyword Score Filter
        if result["keyword_score"] < min_keyword_score:
            continue
        
        # 2. Apply Minimum Semantic Score Filter
        if result["semantic_score"] < min_semantic_score:
            continue

        # 3. Apply Keyword Filters (fuzzy matching for filters too)
        if stemmed_keyword_filters:
            matched_by_filter = False
            for f_kw in stemmed_keyword_filters:
                if any(levenshtein_distance(f_kw, matched_kw) <= LEVENSHTEIN_THRESHOLD for matched_kw in result["matched_keywords"]):
                    matched_by_filter = True
                    break
            if not matched_by_filter:
                continue
        
        # 4. Apply Cancer Type Filter
        if cancer_types and entry["cancer_type"]:
            entry_types = {ct.strip() for ct in entry["cancer_type"].split(",")}
            if not any(ct in entry_types for ct in cancer_types):
                continue
        
        # 5. Apply Gene Filter
        if genes and entry["genes"]:
            entry_genes = {g.strip() for g in entry["genes"].split(",")}
            if not any(g in entry_genes for g in genes):
                continue
        
        filtered.append(result)
    return filtered

# --- UI Helper for Keyword Highlighting ---

def highlight_keywords(text, keywords):
    """
    Highlights (bolds) specified keywords within a given text using Markdown.
    It handles case-insensitivity and ensures whole word matches.
    """
    highlighted_text = text
    # Sort keywords by length descending to prevent shorter words from being bolded within longer ones
    sorted_keywords = sorted(list(keywords), key=len, reverse=True)
    
    for keyword in sorted_keywords:
        # Create a regex pattern to match the whole word (case-insensitive)
        # re.escape handles special characters in keywords
        pattern = r'\b(' + re.escape(keyword) + r')\b'
        # Replace matched word with bolded version
        highlighted_text = re.sub(pattern, r'**\1**', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

# --- Knowledge Map Visualization ---

def create_knowledge_map(query, results, top_n_entities=5):
    """
    Generates a static knowledge map (graph) showing the query and top related
    cancer types and genes from the search results.
    """
    G = nx.Graph()
    G.add_node(query, type='query', color='skyblue', size=500)

    # Collect entity frequencies from results
    cancer_type_counts = defaultdict(int)
    gene_counts = defaultdict(int)

    for res in results:
        if res["entry"]["cancer_type"]:
            for ct in res["entry"]["cancer_type"].split(","):
                cancer_type_counts[ct.strip()] += 1
        if res["entry"]["genes"]:
            for gene in res["entry"]["genes"].split(","):
                gene_counts[gene.strip()] += 1

    # Add top cancer types and genes
    top_cancer_types = sorted(cancer_type_counts.items(), key=lambda item: item[1], reverse=True)[:top_n_entities]
    top_genes = sorted(gene_counts.items(), key=lambda item: item[1], reverse=True)[:top_n_entities]

    node_colors = []
    node_sizes = []
    labels = {}

    # Add query node properties
    labels[query] = query
    node_colors.append('skyblue')
    node_sizes.append(1000) # Larger size for query

    # Add cancer type nodes and edges
    for ct, count in top_cancer_types:
        G.add_node(ct, type='cancer_type', color='lightcoral', size=300 + count*50)
        G.add_edge(query, ct, relation='related_to', weight=count)
        labels[ct] = ct
        node_colors.append('lightcoral')
        node_sizes.append(300 + count*50)

    # Add gene nodes and edges
    for gene, count in top_genes:
        G.add_node(gene, type='gene', color='lightgreen', size=300 + count*50)
        G.add_edge(query, gene, relation='related_to', weight=count)
        labels[gene] = gene
        node_colors.append('lightgreen')
        node_sizes.append(300 + count*50)

    if not G.nodes: # If no nodes were added (e.g., no results)
        return None

    # Draw the graph
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, iterations=50) # Use spring layout for better node distribution

    nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[node]['color'] for node in G.nodes()], node_size=[G.nodes[node]['size'] for node in G.nodes()])
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

    plt.title("Knowledge Map of Search Results", size=15)
    plt.axis('off') # Hide axes
    
    # Save plot to a BytesIO object and encode to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.5)
    plt.close(fig) # Close the figure to free memory
    data_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data_url}"

# --- Streamlit Page Layouts ---

def show_home():
    """Defines the layout and behavior of the application's home page."""
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials** 🔍
    This tool helps researchers access structured clinical trial information.
    """)

    # Display random suggestions to guide the user
    if st.session_state.suggestions:
        st.markdown("---")
        st.markdown("**💡 Try these sample questions:**")
        cols = st.columns(2) # Display suggestions in two columns
        for i, suggestion in enumerate(st.session_state.suggestions[:6]): # Limit to 6 suggestions
            with cols[i % 2]: # Distribute buttons across columns
                # Create a button for each suggestion
                if st.button(suggestion[:70] + "..." if len(suggestion) > 70 else suggestion, 
                             key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion # Set query and navigate to results
                    st.session_state.show_home = False
                    st.rerun() # Rerun the app to switch to results page
        st.markdown("---")

    # Search input form
    with st.form("search_form"):
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query, # Pre-fill with current query if any
            placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high NSCLC patients?",
            help="Enter your clinical question or keywords"
        )

        # Submit button for the search form
        if st.form_submit_button("Search", type="primary"):
            if query.strip(): # Only proceed if query is not empty
                st.session_state.current_query = query.strip()
                st.session_state.show_home = False
                # Add current query to search history (most recent first)
                st.session_state.search_history.insert(0, {
                    "query": st.session_state.current_query,
                    "timestamp": datetime.now().isoformat()
                })
                # Keep history limited to a reasonable number (e.g., 20)
                st.session_state.search_history = st.session_state.search_history[:20]
                save_search_history() # Save history to file
                st.rerun() # Rerun to show results
            else:
                st.warning("Please enter a search query.")

def show_results():
    """Defines the layout and behavior of the search results page."""
    # Button to go back to the home page
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.rerun()

    st.title("🔍 Search Results")
    
    # --- Filters Sidebar ---
    with st.sidebar:
        st.subheader("🔎 Refine Results")
        
        # Slider for Minimum Keyword Match Score
        st.session_state.min_keyword_score = st.slider(
            "Minimum Keyword Match Score",
            min_value=0,
            max_value=20, # Max value adjusted for potential fuzzy scores
            value=st.session_state.min_keyword_score,
            help="Higher scores mean more exact/fuzzy keyword matches"
        )

        # Slider for Minimum Semantic Similarity Score
        st.session_state.min_semantic_score = st.slider(
            "Minimum Semantic Similarity (0.0 - 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.min_semantic_score,
            step=0.05,
            help="Higher scores mean more conceptual similarity. Requires spaCy model."
        )
        
        # Input for adding keyword filters
        st.markdown("**Filter by keywords:**")
        new_keyword = st.text_input("Add keyword filter", key="new_keyword_filter_input")
        add_keyword_button = st.button("Add Keyword Filter", key="add_keyword_filter_button")

        if add_keyword_button and new_keyword.strip():
            # Add keyword only if not already present (case-insensitive)
            if new_keyword.lower() not in [k.lower() for k in st.session_state.keyword_filters]:
                st.session_state.keyword_filters.append(new_keyword.strip())
                st.rerun() # Rerun to apply filter

        # Multiselect for Cancer Type filters
        if st.session_state.cancer_types:
            st.markdown("---")
            st.markdown("**Filter by cancer type:**")
            selected_cancers = st.multiselect(
                "Select cancer types",
                st.session_state.cancer_types,
                default=st.session_state.cancer_type_filter,
                key="cancer_type_select"
            )
            # Update session state and rerun if selection changes
            if selected_cancers != st.session_state.cancer_type_filter:
                st.session_state.cancer_type_filter = selected_cancers
                st.rerun()
        
        # Multiselect for Gene filters
        if st.session_state.genes:
            st.markdown("---")
            st.markdown("**Filter by genes:**")
            selected_genes = st.multiselect(
                "Select genes",
                st.session_state.genes,
                default=st.session_state.gene_filter,
                key="gene_select"
            )
            # Update session state and rerun if selection changes
            if selected_genes != st.session_state.gene_filter:
                st.session_state.gene_filter = selected_genes
                st.rerun()
        
        st.markdown("---")
        # Display active filters and provide 'X' buttons to remove them
        if st.session_state.keyword_filters or st.session_state.cancer_type_filter or st.session_state.gene_filter:
            st.markdown("**Active Filters:**")
            filters_to_remove = []

            # Display and allow removal of keyword filters
            for i, f_kw in enumerate(st.session_state.keyword_filters):
                if st.button(f"❌ Keyword: {f_kw}", key=f"remove_kw_{i}"):
                    filters_to_remove.append(("keyword", f_kw))
            
            # Display and allow removal of cancer type filters
            for i, f_ct in enumerate(st.session_state.cancer_type_filter):
                if st.button(f"❌ Cancer Type: {f_ct}", key=f"remove_ct_{i}"):
                    filters_to_remove.append(("cancer_type", f_ct))

            # Display and allow removal of gene filters
            for i, f_gene in enumerate(st.session_state.gene_filter):
                if st.button(f"❌ Gene: {f_gene}", key=f"remove_gene_{i}"):
                    filters_to_remove.append(("gene", f_gene))

            # Apply removals if any 'X' button was clicked
            if filters_to_remove:
                for f_type, f_value in filters_to_remove:
                    if f_type == "keyword":
                        st.session_state.keyword_filters.remove(f_value)
                    elif f_type == "cancer_type":
                        st.session_state.cancer_type_filter.remove(f_value)
                    elif f_type == "gene":
                        st.session_state.gene_filter.remove(f_value)
                st.rerun() # Rerun to update filters

        # Button to clear all active filters
        if st.button("Clear All Filters", type="secondary"):
            st.session_state.keyword_filters = []
            st.session_state.cancer_type_filter = []
            st.session_state.gene_filter = []
            st.session_state.min_keyword_score = 1 # Reset min score
            st.session_state.min_semantic_score = 0.0 # Reset semantic score
            st.rerun() # Rerun to clear filters

    # --- Main Results Area ---

    # Load data and index (cached for efficiency)
    data, word_index, cancer_types, genes, _ = load_and_index_data()
    if data is None: # Handle case where data loading failed
        return

    # Store cancer types and genes in session state if not already populated
    if not st.session_state.cancer_types and cancer_types:
        st.session_state.cancer_types = cancer_types
    if not st.session_state.genes and genes:
        st.session_state.genes = genes

    # Display current search query
    st.markdown(f"**Current Search Query:** **`{st.session_state.current_query}`**")

    # Perform search and filtering with a spinner for user feedback
    with st.spinner("Searching clinical knowledge base..."):
        ranked_results = combined_search(st.session_state.current_query, data, word_index, nlp)
        filtered_results = filter_results(
            ranked_results,
            st.session_state.min_keyword_score,
            st.session_state.min_semantic_score,
            st.session_state.keyword_filters,
            st.session_state.cancer_type_filter,
            st.session_state.gene_filter
        ) if ranked_results else [] # Ensure ranked_results is not empty before filtering

        if filtered_results:
            display_results(filtered_results, ranked_results)
        else:
            show_no_results(data, word_index)

def display_results(results, all_results):
    """
    Displays the filtered search results, including download options, match details,
    and the knowledge map.
    """
    st.success(f"Found {len(results)} relevant results (from {len(all_results)} total matches before filtering)")
    
    # --- Knowledge Map Display ---
    st.markdown("---")
    st.subheader("🌐 Knowledge Map for Current Search")
    if results:
        with st.spinner("Generating knowledge map..."):
            knowledge_map_url = create_knowledge_map(st.session_state.current_query, results)
            if knowledge_map_url:
                st.image(knowledge_map_url, caption="Connections based on search results", use_column_width=True)
            else:
                st.info("Not enough data to generate a meaningful knowledge map for these results.")
    else:
        st.info("No results to generate a knowledge map. Try broadening your search.")
    st.markdown("---")

    # Score distribution chart
    if len(results) > 1:
        scores = [r["total_score"] for r in results]
        st.bar_chart(pd.DataFrame({"Total Match Score": scores}), use_container_width=True)
    
    # Download buttons for all currently filtered results (JSON and CSV)
    all_results_data = [result["entry"] for result in results]
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Filtered Results as JSON",
            json.dumps(all_results_data, indent=2),
            file_name="filtered_cancer_search_results.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            "Download Filtered Results as CSV",
            pd.DataFrame(all_results_data).to_csv(index=False),
            file_name="filtered_cancer_search_results.csv",
            mime="text/csv"
        )

    # Display each individual result in an expander
    for i, result in enumerate(results, 1):
        entry = result["entry"]
        
        # Prepare keywords for highlighting: original query words (stemmed) + actual matched words from the entry
        highlight_terms = set(stemmer.stem(word) for word in st.session_state.current_query.lower().split() if word.strip())
        highlight_terms.update(result["matched_keywords"]) # Add actual matched keywords from entry

        with st.expander(f"#{i} | Total Score: {result['total_score']:.2f} - {entry['prompt'][:70]}...", expanded=(i==1)):
            st.markdown(f"**Question:** {highlight_keywords(entry['prompt'], highlight_terms)}")
            st.markdown(f"**Answer:** {highlight_keywords(entry['completion'], highlight_terms)}")
            
            # Display metadata if available
            metadata_parts = []
            if entry["cancer_type"]:
                metadata_parts.append(f"**Cancer Type:** {entry['cancer_type']}")
            if entry["genes"]:
                metadata_parts.append(f"**Genes:** {entry['genes']}")
            if metadata_parts:
                st.markdown(" | ".join(metadata_parts))
            
            # Detailed match information in a nested expander
            with st.expander("🔍 Match Details"):
                st.markdown(f"**Keyword Score:** {result['keyword_score']:.2f}")
                st.markdown(f"**Semantic Similarity:** {result['semantic_score']:.2f}")
                st.markdown(f"**Total Combined Score:** {result['total_score']:.2f}")
                st.markdown(f"**Prompt Keyword Matches:** {result['prompt_matches']:.2f}")
                st.markdown(f"**Answer Keyword Matches:** {result['completion_matches']:.2f}")
                if result["matched_keywords"]:
                    # Display the stemmed keywords that contributed to the match
                    st.markdown(f"**Keywords Involved in Match:** `{', '.join(result['matched_keywords'])}`")
            
            # Download buttons for individual result
            col1_dl, col2_dl = st.columns(2)
            with col1_dl:
                st.download_button(
                    "Download as JSON",
                    json.dumps(entry, indent=2),
                    file_name=f"cancer_result_{i}.json",
                    mime="application/json",
                    key=f"json_single_{i}" # Unique key for each button
                )
            with col2_dl:
                st.download_button(
                    "Download as CSV",
                    pd.DataFrame([entry]).to_csv(index=False),
                    file_name=f"cancer_result_{i}.csv",
                    mime="text/csv",
                    key=f"csv_single_{i}" # Unique key for each button
                )

def show_no_results(data, word_index):
    """
    Displays a message when no results are found and provides search tips and suggestions.
    """
    st.error("😞 No matches found with current filters.")
    st.info("Try these suggestions or adjust your filters:")
    
    # Generate suggestions based on the current query's stemmed words, including fuzzy matches
    query_words_stemmed = get_clean_words(st.session_state.current_query)
    suggestions_from_query = set()

    if query_words_stemmed and word_index and data:
        potential_doc_ids = set()
        # Collect document IDs that contain any of the query words (stemmed)
        for q_word in query_words_stemmed:
            if q_word in word_index:
                potential_doc_ids.update(word_index[q_word])
        
        # Also consider fuzzy matches to suggest more relevant prompts
        all_indexed_words = list(word_index.keys())
        for q_word in query_words_stemmed:
            for indexed_word in all_indexed_words:
                if levenshtein_distance(q_word, indexed_word) <= LEVENSHTEIN_THRESHOLD:
                    # Add documents associated with fuzzy matched words
                    potential_doc_ids.update(word_index[indexed_word])
        
        # Limit suggestions to a reasonable number of unique prompts
        for doc_id in list(potential_doc_ids)[:20]: # Check up to 20 potential documents
            if doc_id < len(data):
                suggestions_from_query.add(data[doc_id]["prompt"])
                if len(suggestions_from_query) >= 5: # Show up to 5 unique suggestions
                    break

    if suggestions_from_query:
        st.write("**You might find these related questions helpful:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(list(suggestions_from_query)):
            with cols[i % 2]:
                if st.button(suggestion[:70] + "..." if len(suggestion) > 70 else suggestion, 
                             key=f"nores_sugg_{i}"):
                    st.session_state.current_query = suggestion
                    st.session_state.show_home = False
                    st.rerun()
    else:
        st.write("**No immediate suggestions found based on your query.**")
        st.markdown("Try broadening your search terms or removing some filters.")

    st.markdown("---")
    st.markdown("""
    **General Search Tips:**
    - Try lowering the **Minimum Keyword Match Score** or **Minimum Semantic Similarity** filters.
    - Remove some **Keyword**, **Cancer Type**, or **Gene** filters to broaden your search.
    - Double-check for typos in your search terms.
    - Use more general terms if your search is too specific.
    - Consider alternative phrasing for your question.
    """)

# --- Main Application Flow ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="🧬 Cancer Clinical Trial Search",
        page_icon="🧬",
        layout="wide", # Use wide layout for better display of results and filters
        initial_sidebar_state="expanded" # Sidebar expanded by default
    )

    # Load search history at the start of the app
    if not st.session_state.search_history:
        st.session_state.search_history = load_search_history()

    # Load data and suggestions only once or if not already loaded (cached)
    # This populates the multiselect options for filters
    if not st.session_state.suggestions or not st.session_state.cancer_types or not st.session_state.genes:
        # Ensure spaCy model is loaded before data
        _ = nlp # This will trigger load_spacy_model if not already loaded
        data, word_index, cancer_types, genes, suggestions = load_and_index_data()
        if suggestions:
            st.session_state.suggestions = suggestions
        if cancer_types:
            st.session_state.cancer_types = cancer_types
        if genes:
            st.session_state.genes = genes

    # Sidebar content (always visible)
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Cancer+Search+MVP", width=150) # Placeholder image
        st.title("Navigation")

        # "Home" button only appears when not on the home page
        if not st.session_state.show_home:
            if st.button("🏠 Home"):
                st.session_state.show_home = True
                st.rerun()

        st.markdown("---")
        st.markdown("### About This App")
        st.markdown("""
        This **MVP (Minimum Viable Product)** allows researchers to quickly **extract and filter** information from **JSON-formatted clinical trial data**. 
        It supports:
        - **Keyword-based search** with enhanced fuzzy matching.
        - **Semantic search** using pre-trained word embeddings for conceptual understanding.
        - Filtering by **cancer types** and **genes**.
        - Viewing and downloading **structured data**.
        - A **search history** for easy recall of past queries.
        - A **knowledge map** to visualize relationships in search results.
        
        The app uses a basic text processing pipeline including **stemming** and **stop word removal** for more effective search results.
        """)

    # Main content area: switch between home and results pages
    if st.session_state.show_home:
        show_home()
    else:
        show_results()

# Entry point of the Streamlit application
if __name__ == "__main__":
    main()

