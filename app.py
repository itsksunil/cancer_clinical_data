import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os

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

# Constants
DATA_FILE = "cancer_clinical_dataset.json"
HISTORY_FILE = "search_history.json"

# Load and save search history
def load_search_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except:
        return []

def save_search_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {str(e)}")

# Validate data file
def validate_data_file():
    if not os.path.exists(DATA_FILE):
        st.error(f"Critical Error: Data file '{DATA_FILE}' not found.")
        st.stop()
    
    if os.path.getsize(DATA_FILE) == 0:
        st.error(f"Critical Error: Data file '{DATA_FILE}' is empty.")
        st.stop()
    
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Critical Error: Invalid JSON in data file: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Critical Error: Cannot read data file: {str(e)}")
        st.stop()

# Load and preprocess data with caching
@st.cache_data
def load_and_index_data():
    try:
        validate_data_file()
        
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        clean_data = []
        word_index = defaultdict(list)
        cancer_types = set()
        genes = set()
        all_prompts = []

        if not isinstance(raw_data, list):
            st.error("Invalid data format: Expected a list of entries.")
            return None, None, [], [], []

        for idx, entry in enumerate(raw_data):
            if not isinstance(entry, dict):
                st.error(f"Entry {idx} is not a dictionary, skipping...")
                continue
                
            if "prompt" not in entry or "completion" not in entry:
                st.error(f"Entry {idx} missing required fields (prompt/completion), skipping...")
                continue
                
            # Create cleaned entry
            clean_entry = {
                "prompt": str(entry["prompt"]).strip(),
                "completion": str(entry["completion"]).strip(),
                "cancer_type": "",
                "genes": "",
                "nct_id": entry.get("nct_id", "")
            }
            
            # Extract cancer type from prompt/completion
            cancer_keywords = {
                "breast cancer": "Breast Cancer",
                "nsclc": "NSCLC",
                "lung cancer": "Lung Cancer",
                "colorectal": "Colorectal Cancer"
            }
            
            for kw, cancer_type in cancer_keywords.items():
                if kw in clean_entry["prompt"].lower() or kw in clean_entry["completion"].lower():
                    clean_entry["cancer_type"] = cancer_type
                    cancer_types.add(cancer_type)
                    break
                    
            # Extract genes from prompt/completion
            gene_keywords = {
                "er+": "ESR1",
                "her2-": "HER2",
                "egfr": "EGFR",
                "alk": "ALK"
            }
            
            found_genes = []
            for kw, gene in gene_keywords.items():
                if kw in clean_entry["prompt"].lower() or kw in clean_entry["completion"].lower():
                    found_genes.append(gene)
                    
            if found_genes:
                clean_entry["genes"] = ", ".join(found_genes)
                genes.update(found_genes)
                
            # Add to clean data
            clean_data.append(clean_entry)
            all_prompts.append(clean_entry["prompt"])
            
            # Index words
            for field in ["prompt", "completion"]:
                for word in set(clean_entry[field].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None, [], [], []
        
        # Generate random suggestions
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions
    
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return None, None, [], [], []

# Enhanced keyword search with filters
def keyword_search(query, dataset, word_index):
    if not query or not dataset:
        return []
    
    query_words = set(word.lower() for word in query.split() if len(word) > 2)
    
    doc_matches = defaultdict(int)
    for word in query_words:
        if word in word_index:
            for doc_id in word_index[word]:
                if doc_id < len(dataset):
                    doc_matches[doc_id] += 1

    ranked_results = []
    for doc_id, count in doc_matches.items():
        entry = dataset[doc_id]
        prompt_words = set(entry["prompt"].lower().split())
        completion_words = set(entry["completion"].lower().split())

        prompt_matches = len(query_words & prompt_words)
        completion_matches = len(query_words & completion_words)
        total_score = (prompt_matches * 2) + completion_matches

        ranked_results.append({
            "entry": entry,
            "score": total_score,
            "prompt_matches": prompt_matches,
            "completion_matches": completion_matches,
            "matched_keywords": query_words & (prompt_words | completion_words)
        })

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results

# Filter results based on multiple criteria
def filter_results(results, min_score, keyword_filters, cancer_types, genes):
    filtered = []
    for result in results:
        entry = result["entry"]
        
        # Apply score filter
        if result["score"] < min_score:
            continue
        
        # Apply keyword filters
        if keyword_filters:
            matched = any(kw.lower() in result["matched_keywords"] for kw in keyword_filters)
            if not matched:
                continue
        
        # Apply cancer type filter
        if cancer_types and entry["cancer_type"]:
            entry_types = set(ct.strip() for ct in entry["cancer_type"].split(","))
            if not any(ct in entry_types for ct in cancer_types):
                continue
        
        # Apply gene filter
        if genes and entry["genes"]:
            entry_genes = set(g.strip() for g in entry["genes"].split(","))
            if not any(g in entry_genes for g in genes):
                continue
        
        filtered.append(result)
    return filtered

# Home page layout
def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials**  
    This tool helps researchers access structured clinical trial information.
    """)

    # Display random suggestions
    if st.session_state.suggestions:
        st.markdown("**💡 Try these sample questions:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(st.session_state.suggestions[:6]):
            with cols[i % 2]:
                if st.button(suggestion[:50] + "..." if len(suggestion) > 50 else suggestion, 
                            key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.session_state.show_home = False
                    st.rerun()

    with st.form("search_form"):
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query,
            placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high NSCLC patients?",
            help="Enter your clinical question or keywords"
        )

        if st.form_submit_button("Search", type="primary") and query.strip():
            st.session_state.current_query = query
            st.session_state.show_home = False
            st.session_state.search_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat()
            })
            save_search_history()
            st.rerun()

# Results page layout
def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.rerun()

    st.title("🔍 Search Results")
    
    # Filters sidebar
    with st.sidebar:
        st.subheader("🔎 Refine Results")
        
        # Score filter
        st.session_state.min_score = st.slider(
            "Minimum Match Score",
            min_value=0,
            max_value=20,
            value=1,
            help="Higher scores mean more keywords matched"
        )
        
        # Keyword filters
        st.markdown("**Filter by keywords:**")
        new_keyword = st.text_input("Add keyword filter", key="new_keyword")
        if st.button("Add Keyword Filter") and new_keyword.strip():
            if new_keyword.lower() not in [k.lower() for k in st.session_state.keyword_filters]:
                st.session_state.keyword_filters.append(new_keyword.strip())
                st.rerun()
        
        # Cancer type filter
        if st.session_state.cancer_types:
            st.markdown("**Filter by cancer type:**")
            selected_cancers = st.multiselect(
                "Select cancer types",
                st.session_state.cancer_types,
                default=st.session_state.cancer_type_filter,
                key="cancer_type_select"
            )
            if selected_cancers != st.session_state.cancer_type_filter:
                st.session_state.cancer_type_filter = selected_cancers
                st.rerun()
        
        # Gene filter
        if st.session_state.genes:
            st.markdown("**Filter by genes:**")
            selected_genes = st.multiselect(
                "Select genes",
                st.session_state.genes,
                default=st.session_state.gene_filter,
                key="gene_select"
            )
            if selected_genes != st.session_state.gene_filter:
                st.session_state.gene_filter = selected_genes
                st.rerun()
        
        # Display active filters
        active_filters = []
        if st.session_state.keyword_filters:
            active_filters.extend([f"Keyword: {kw}" for kw in st.session_state.keyword_filters])
        if st.session_state.cancer_type_filter:
            active_filters.extend([f"Cancer: {ct}" for ct in st.session_state.cancer_type_filter])
        if st.session_state.gene_filter:
            active_filters.extend([f"Gene: {g}" for g in st.session_state.gene_filter])
        
        if active_filters:
            st.markdown("**Active Filters:**")
            for i, f in enumerate(active_filters):
                cols = st.columns([1, 4])
                with cols[0]:
                    if st.button("❌", key=f"remove_{i}"):
                        if f.startswith("Keyword:"):
                            st.session_state.keyword_filters.remove(f[8:])
                        elif f.startswith("Cancer:"):
                            st.session_state.cancer_type_filter.remove(f[7:])
                        elif f.startswith("Gene:"):
                            st.session_state.gene_filter.remove(f[5:])
                        st.rerun()
                with cols[1]:
                    st.markdown(f)
        
        if st.button("Clear All Filters"):
            st.session_state.keyword_filters = []
            st.session_state.cancer_type_filter = []
            st.session_state.gene_filter = []
            st.session_state.min_score = 1
            st.rerun()

    # Load data
    data, word_index, cancer_types, genes, _ = load_and_index_data()
    if data is None:
        return

    # Store cancer types and genes in session state
    if cancer_types:
        st.session_state.cancer_types = cancer_types
    if genes:
        st.session_state.genes = genes

    # Display search history
    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            history_cols = st.columns(2)
            for i, search in enumerate(reversed(st.session_state.search_history)):
                with history_cols[i % 2]:
                    if st.button(f"{search['query'][:50]}{'...' if len(search['query']) > 50 else ''}",
                               key=f"history_{i}"):
                        st.session_state.current_query = search["query"]
                        st.rerun()

    st.markdown(f"**Current Search:** {st.session_state.current_query}")

    with st.spinner("Searching clinical knowledge base..."):
        ranked_results = keyword_search(st.session_state.current_query, data, word_index)
        filtered_results = filter_results(
            ranked_results,
            st.session_state.min_score,
            st.session_state.keyword_filters,
            st.session_state.cancer_type_filter,
            st.session_state.gene_filter
        ) if ranked_results else []

        if filtered_results:
            display_results(filtered_results, ranked_results)
        else:
            show_no_results(data, word_index)

def display_results(results, all_results):
    st.success(f"Found {len(results)} relevant results (from {len(all_results)} total matches)")
    
    # Score distribution chart
    if len(results) > 1:
        scores = [r["score"] for r in results]
        st.bar_chart(pd.DataFrame({"Score": scores}), use_container_width=True)
    
    # Download buttons
    all_results_data = [result["entry"] for result in results]
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download All as JSON",
            json.dumps(all_results_data, indent=2),
            file_name="cancer_search_results.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            "Download All as CSV",
            pd.DataFrame(all_results_data).to_csv(index=False),
            file_name="cancer_search_results.csv",
            mime="text/csv"
        )

    # Display results
    for i, result in enumerate(results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']} - {entry['prompt'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Question:** {entry['prompt']}")
            st.markdown(f"**Answer:** {entry['completion']}")
            
            # Metadata
            metadata = []
            if entry.get("nct_id"):
                metadata.append(f"**ClinicalTrials.gov ID:** {entry['nct_id']}")
            if entry["cancer_type"]:
                metadata.append(f"**Cancer Type:** {entry['cancer_type']}")
            if entry["genes"]:
                metadata.append(f"**Genes:** {entry['genes']}")
            if metadata:
                st.markdown(" | ".join(metadata))
            
            # Score details
            with st.expander("🔍 Match Details"):
                st.markdown(f"**Total Score:** {result['score']}")
                st.markdown(f"**Prompt Matches:** {result['prompt_matches']}")
                st.markdown(f"**Answer Matches:** {result['completion_matches']}")
                if result["matched_keywords"]:
                    st.markdown(f"**Matched Keywords:** {', '.join(result['matched_keywords'])}")
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download as JSON",
                    json.dumps(entry, indent=2),
                    file_name=f"cancer_result_{i}.json",
                    mime="application/json",
                    key=f"json_{i}"
                )
            with col2:
                st.download_button(
                    "Download as CSV",
                    pd.DataFrame([entry]).to_csv(index=False),
                    file_name=f"cancer_result_{i}.csv",
                    mime="text/csv",
                    key=f"csv_{i}"
                )

def show_no_results(data, word_index):
    st.error("No matches found with current filters. Try these suggestions:")
    
    # Generate suggestions from query
    query_words = set(word.lower() for word in st.session_state.current_query.split() if len(word) > 3)
    suggestions = set()

    if query_words and word_index and data:
        doc_ids = set()
        for word in query_words:
            if word in word_index:
                doc_ids.update(word_index[word])

        for doc_id in list(doc_ids)[:50]:
            if doc_id < len(data):
                suggestions.add(data[doc_id]["prompt"])
                if len(suggestions) >= 5:
                    break

    if suggestions:
        st.write("**Similar questions in our database:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion[:50] + "..." if len(suggestion) > 50 else suggestion, 
                            key=f"nores_sugg_{i}"):
                    st.session_state.current_query = suggestion
                    st.rerun()
    
    st.markdown("""
    **Search Tips:**
    - Try lowering the minimum score filter
    - Remove some filters to broaden your search
    - Check for typos in your search terms
    - Use more general terms if your search is too specific
    """)

# Main app flow
def main():
    st.set_page_config(
        page_title="🧬 Cancer Clinical Trial Search",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load search history
    if not st.session_state.search_history:
        st.session_state.search_history = load_search_history()

    # Load data and suggestions
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

        if not st.session_state.show_home:
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

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
