import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os
from difflib import get_close_matches
from typing import List, Dict, Set, Optional

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
if 'all_keywords' not in st.session_state:
    st.session_state.all_keywords = set()
if 'show_advanced_filters' not in st.session_state:
    st.session_state.show_advanced_filters = False

# Constants
DATA_FILE = "cancer_clinical_dataset.json"
HISTORY_FILE = "search_history.json"

# Load and save search history
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

# Enhanced data loading with better indexing
@st.cache_data
def load_and_index_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        clean_data = []
        word_index = defaultdict(list)
        cancer_types = set()
        genes = set()
        all_prompts = []
        all_keywords = set()

        for idx, entry in enumerate(raw_data):
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                # Clean and standardize data
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                
                # Extract metadata
                entry_cancer_types = []
                if "cancer_type" in entry:
                    entry_cancer_types = [ct.strip().title() for ct in str(entry["cancer_type"]).split(",") if ct.strip()]
                    cancer_types.update(entry_cancer_types)
                
                entry_genes = []
                if "genes" in entry:
                    entry_genes = [g.strip().upper() for g in str(entry["genes"]).split(",") if g.strip()]
                    genes.update(entry_genes)
                
                # Store cleaned entry
                clean_data.append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "cancer_type": ", ".join(entry_cancer_types) if entry_cancer_types else "",
                    "genes": ", ".join(entry_genes) if entry_genes else "",
                    "id": idx
                })
                all_prompts.append(entry["prompt"])

                # Index words with stemming and better tokenization
                text = f"{entry['prompt']} {entry['completion']}".lower()
                words = set()
                for word in text.split():
                    # Basic stemming - remove common suffixes
                    word = word.strip(".,!?;:-_'\"()[]{}")
                    if len(word) > 2 and word.isalpha():
                        words.add(word)
                        all_keywords.add(word)
                
                for word in words:
                    word_index[word].append(idx)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None, [], [], [], []
        
        # Generate random suggestions
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions, sorted(all_keywords)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, [], [], [], []

# Enhanced search with fuzzy matching and phrase detection
def keyword_search(query: str, dataset: List[Dict], word_index: Dict[str, List[int]]) -> List[Dict]:
    if not query or not dataset:
        return []
    
    # Preprocess query
    query = query.lower()
    query_words = set(word.strip(".,!?;:-_'\"()[]{}") for word in query.split() if len(word) > 2 and word.isalpha())
    
    # Find similar words for fuzzy matching
    expanded_words = set(query_words)
    for word in query_words:
        similar_words = get_close_matches(word, st.session_state.all_keywords, n=3, cutoff=0.7)
        expanded_words.update(similar_words)
    
    # Find documents containing any of the words
    doc_matches = defaultdict(int)
    doc_word_matches = defaultdict(set)
    
    for word in expanded_words:
        if word in word_index:
            for doc_id in word_index[word]:
                if doc_id < len(dataset):
                    doc_matches[doc_id] += 1
                    doc_word_matches[doc_id].add(word)
    
    # Rank results
    ranked_results = []
    for doc_id, count in doc_matches.items():
        entry = dataset[doc_id]
        full_text = f"{entry['prompt']} {entry['completion']}".lower()
        
        # Calculate exact matches
        exact_matches = sum(1 for word in query_words if word in full_text)
        
        # Phrase detection - check if multiple query words appear together
        phrase_bonus = 0
        for i in range(len(query.split()) - 1):
            phrase = " ".join(query.split()[i:i+2]).lower()
            if phrase in full_text:
                phrase_bonus += 3
        
        # Position bonus - matches in prompt are more valuable
        prompt_matches = sum(1 for word in query_words if word in entry["prompt"].lower())
        
        total_score = (exact_matches * 2) + phrase_bonus + (prompt_matches * 1.5)
        
        ranked_results.append({
            "entry": entry,
            "score": total_score,
            "exact_matches": exact_matches,
            "phrase_bonus": phrase_bonus,
            "prompt_matches": prompt_matches,
            "matched_keywords": doc_word_matches[doc_id],
            "doc_id": doc_id
        })

    # Sort by score and then by number of exact matches
    ranked_results.sort(key=lambda x: (-x["score"], -x["exact_matches"]))
    return ranked_results

# Enhanced filter system
def filter_results(results: List[Dict], min_score: int, keyword_filters: List[str], 
                  cancer_types: List[str], genes: List[str]) -> List[Dict]:
    filtered = []
    for result in results:
        entry = result["entry"]
        
        # Score filter
        if result["score"] < min_score:
            continue
        
        # Keyword filters
        if keyword_filters:
            text = f"{entry['prompt']} {entry['completion']}".lower()
            if not all(kw.lower() in text for kw in keyword_filters):
                continue
        
        # Cancer type filter
        if cancer_types and entry["cancer_type"]:
            entry_types = set(ct.strip().title() for ct in entry["cancer_type"].split(","))
            if not any(ct in entry_types for ct in cancer_types):
                continue
        
        # Gene filter
        if genes and entry["genes"]:
            entry_genes = set(g.strip().upper() for g in entry["genes"].split(","))
            if not any(g in entry_genes for g in genes):
                continue
        
        filtered.append(result)
    return filtered

# Auto-suggest function
def get_suggestions(partial_query: str, all_keywords: Set[str]) -> List[str]:
    if not partial_query or len(partial_query) < 2:
        return []
    
    partial = partial_query.lower()
    suggestions = []
    
    # Match beginning of words
    suggestions.extend([kw for kw in all_keywords if kw.startswith(partial)])
    
    # Fuzzy matches
    suggestions.extend(get_close_matches(partial, all_keywords, n=5, cutoff=0.6))
    
    # Remove duplicates and limit results
    return list(set(suggestions))[:10]

# Home page with enhanced search
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
        # Search box with auto-suggest
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query,
            placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high NSCLC patients?",
            help="Enter your clinical question or keywords",
            key="search_input"
        )
        
        # Show auto-suggestions
        if query and len(query) > 1 and st.session_state.all_keywords:
            suggestions = get_suggestions(query, st.session_state.all_keywords)
            if suggestions:
                st.markdown("**Suggestions:**")
                cols = st.columns(3)
                for i, suggestion in enumerate(suggestions[:3]):
                    with cols[i % 3]:
                        if st.button(suggestion, key=f"sugg_{i}"):
                            st.session_state.current_query = suggestion
                            st.rerun()

        # Advanced search options
        with st.expander("🔍 Advanced Search Options", expanded=False):
            st.markdown("**Search in:**")
            search_in = st.radio(
                "Search scope",
                ["Both questions and answers", "Questions only", "Answers only"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown("**Match type:**")
            match_type = st.radio(
                "Matching strategy",
                ["All words (AND)", "Any word (OR)", "Exact phrase"],
                horizontal=True,
                label_visibility="collapsed"
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

# Enhanced results page
def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.rerun()

    st.title("🔍 Search Results")
    
    # Enhanced filters sidebar
    with st.sidebar:
        st.subheader("🔎 Refine Results")
        
        # Score filter with distribution info
        st.session_state.min_score = st.slider(
            "Minimum Match Score",
            min_value=0,
            max_value=20,
            value=1,
            help="Higher scores mean more keywords matched"
        )
        
        # Keyword filters with better UI
        st.markdown("**Filter by keywords:**")
        keyword_cols = st.columns([4, 1])
        with keyword_cols[0]:
            new_keyword = st.text_input("Add keyword filter", key="new_keyword")
        with keyword_cols[1]:
            if st.button("Add", key="add_keyword") and new_keyword.strip():
                if new_keyword.lower() not in [k.lower() for k in st.session_state.keyword_filters]:
                    st.session_state.keyword_filters.append(new_keyword.strip())
                    st.rerun()
        
        # Show current keyword filters
        if st.session_state.keyword_filters:
            st.markdown("**Active keyword filters:**")
            for i, kw in enumerate(st.session_state.keyword_filters):
                cols = st.columns([1, 4])
                with cols[0]:
                    if st.button("❌", key=f"remove_kw_{i}"):
                        st.session_state.keyword_filters.pop(i)
                        st.rerun()
                with cols[1]:
                    st.markdown(kw)
        
        # Cancer type filter with search
        if st.session_state.cancer_types:
            st.markdown("**Filter by cancer type:**")
            cancer_search = st.text_input("Search cancer types", key="cancer_search")
            
            if cancer_search:
                cancer_options = [ct for ct in st.session_state.cancer_types 
                                if cancer_search.lower() in ct.lower()]
            else:
                cancer_options = st.session_state.cancer_types
            
            selected_cancers = st.multiselect(
                "Select cancer types",
                cancer_options,
                default=st.session_state.cancer_type_filter,
                key="cancer_type_select",
                label_visibility="collapsed"
            )
            if selected_cancers != st.session_state.cancer_type_filter:
                st.session_state.cancer_type_filter = selected_cancers
                st.rerun()
        
        # Gene filter with search
        if st.session_state.genes:
            st.markdown("**Filter by genes:**")
            gene_search = st.text_input("Search genes", key="gene_search")
            
            if gene_search:
                gene_options = [g for g in st.session_state.genes 
                              if gene_search.upper() in g.upper()]
            else:
                gene_options = st.session_state.genes
            
            selected_genes = st.multiselect(
                "Select genes",
                gene_options,
                default=st.session_state.gene_filter,
                key="gene_select",
                label_visibility="collapsed"
            )
            if selected_genes != st.session_state.gene_filter:
                st.session_state.gene_filter = selected_genes
                st.rerun()
        
        # Clear all filters button
        if st.button("Clear All Filters", type="secondary"):
            st.session_state.keyword_filters = []
            st.session_state.cancer_type_filter = []
            st.session_state.gene_filter = []
            st.session_state.min_score = 1
            st.rerun()

    # Load data
    data, word_index, cancer_types, genes, _ = load_and_index_data()
    if data is None:
        return

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

    st.markdown(f"**Current Search:** `{st.session_state.current_query}`")
    
    # Show active filters
    active_filters = []
    if st.session_state.min_score > 1:
        active_filters.append(f"Min Score: {st.session_state.min_score}")
    if st.session_state.keyword_filters:
        active_filters.append(f"Keywords: {', '.join(st.session_state.keyword_filters)}")
    if st.session_state.cancer_type_filter:
        active_filters.append(f"Cancer Types: {', '.join(st.session_state.cancer_type_filter)}")
    if st.session_state.gene_filter:
        active_filters.append(f"Genes: {', '.join(st.session_state.gene_filter)}")
    
    if active_filters:
        st.markdown(f"**Active Filters:** `{' | '.join(active_filters)}`")

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

def display_results(results: List[Dict], all_results: List[Dict]):
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

    # Display results with highlighting
    for i, result in enumerate(results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']} - {entry['prompt'][:50]}...", expanded=(i==1)):
            # Highlight matches in prompt
            prompt_display = entry["prompt"]
            for word in result["matched_keywords"]:
                prompt_display = prompt_display.replace(word, f"**{word}**")
            
            st.markdown(f"**Question:** {prompt_display}")
            
            # Highlight matches in answer
            completion_display = entry["completion"]
            for word in result["matched_keywords"]:
                completion_display = completion_display.replace(word, f"**{word}**")
            
            st.markdown(f"**Answer:** {completion_display}")
            
            # Metadata
            metadata = []
            if entry["cancer_type"]:
                metadata.append(f"**Cancer Type:** {entry['cancer_type']}")
            if entry["genes"]:
                metadata.append(f"**Genes:** {entry['genes']}")
            if metadata:
                st.markdown(" | ".join(metadata))
            
            # Score details
            with st.expander("🔍 Match Details"):
                st.markdown(f"**Total Score:** {result['score']}")
                st.markdown(f"**Exact Matches:** {result['exact_matches']}")
                st.markdown(f"**Phrase Bonus:** {result['phrase_bonus']}")
                st.markdown(f"**Prompt Matches:** {result['prompt_matches']}")
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

def show_no_results(data: List[Dict], word_index: Dict[str, List[int]]):
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
    - Try different combinations of keywords
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
        data, word_index, cancer_types, genes, suggestions, all_keywords = load_and_index_data()
        if suggestions:
            st.session_state.suggestions = suggestions
        if cancer_types:
            st.session_state.cancer_types = cancer_types
        if genes:
            st.session_state.genes = genes
        if all_keywords:
            st.session_state.all_keywords = set(all_keywords)

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
