import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os
from difflib import get_close_matches

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

def load_search_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            return json.loads(content) if content.strip() else []
    except Exception:
        return []

def save_search_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {str(e)}")

def validate_data_file():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file '{DATA_FILE}' not found.")
        st.stop()
    if os.path.getsize(DATA_FILE) == 0:
        st.error(f"Data file '{DATA_FILE}' is empty.")
        st.stop()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

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
        skipped_entries = 0

        if not isinstance(raw_data, list):
            st.error("Expected a list of entries in JSON.")
            return None, None, [], [], []

        for idx, entry in enumerate(raw_data):
            if not isinstance(entry, dict):
                skipped_entries += 1
                continue
                
            prompt = str(entry.get("prompt", "")).strip()
            completion = str(entry.get("completion", "")).strip()
            
            if not prompt and not completion:
                skipped_entries += 1
                continue
                
            clean_entry = {
                "prompt": prompt,
                "completion": completion,
                "cancer_type": "",
                "genes": "",
                "nct_id": entry.get("nct_id", ""),
                "other_data": {k: v for k, v in entry.items() 
                             if k not in ["prompt", "completion", "cancer_type", "genes", "nct_id"]}
            }
            
            # Extract cancer type
            if entry.get("cancer_type"):
                clean_entry["cancer_type"] = str(entry["cancer_type"]).strip()
            else:
                cancer_keywords = {
                    "breast cancer": "Breast Cancer",
                    "nsclc": "NSCLC",
                    "lung cancer": "Lung Cancer",
                    "colorectal": "Colorectal Cancer"
                }
                text = f"{prompt} {completion}".lower()
                for kw, cancer_type in cancer_keywords.items():
                    if kw in text:
                        clean_entry["cancer_type"] = cancer_type
                        break
            
            if clean_entry["cancer_type"]:
                cancer_types.add(clean_entry["cancer_type"])
                
            # Extract genes
            if entry.get("genes"):
                clean_entry["genes"] = str(entry["genes"]).strip()
            else:
                gene_keywords = {
                    "er+": "ESR1",
                    "her2-": "HER2",
                    "egfr": "EGFR",
                    "alk": "ALK"
                }
                text = f"{prompt} {completion}".lower()
                found_genes = [gene for kw, gene in gene_keywords.items() if kw in text]
                if found_genes:
                    clean_entry["genes"] = ", ".join(found_genes)
            
            if clean_entry["genes"]:
                genes.update(g.strip() for g in clean_entry["genes"].split(","))
                
            clean_data.append(clean_entry)
            if clean_entry["prompt"]:
                all_prompts.append(clean_entry["prompt"])
            
            # Index words
            for text in [prompt, completion]:
                for word in set(text.lower().split()):
                    if len(word) > 2 and word.isalpha():
                        word_index[word].append(idx)

        if skipped_entries > 0:
            st.warning(f"Skipped {skipped_entries} invalid entries")
            
        if not clean_data:
            st.error("No valid entries found.")
            return None, None, [], [], []
        
        valid_prompts = [p for p in all_prompts if p.strip()]
        random_suggestions = random.sample(valid_prompts, min(10, len(valid_prompts))) if valid_prompts else []
        
        return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions
    
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None, None, [], [], []

def get_search_suggestions(query, word_index):
    query_words = query.split()
    suggestions = []
    for word in query_words:
        if len(word) > 3:
            closest = get_close_matches(word.lower(), word_index.keys(), n=1, cutoff=0.7)
            if closest and closest[0] != word.lower():
                suggestions.append(closest[0])
    return suggestions

def keyword_search(query, dataset, word_index):
    if not query or not dataset:
        return []
    
    query_words = {word.lower() for word in query.split() if len(word) > 2}
    doc_matches = defaultdict(int)
    
    # First pass: exact matches
    for word in query_words:
        if word in word_index:
            for doc_id in word_index[word]:
                if doc_id < len(dataset):
                    doc_matches[doc_id] += 1

    # Second pass: partial matches (if we don't have enough results)
    if len(doc_matches) < 5:
        for word in query_words:
            for dict_word in word_index.keys():
                if word in dict_word:  # partial match
                    for doc_id in word_index[dict_word]:
                        if doc_id < len(dataset):
                            doc_matches[doc_id] += 0.5  # partial match score

    ranked_results = []
    for doc_id, count in doc_matches.items():
        entry = dataset[doc_id]
        prompt_text = entry["prompt"].lower()
        completion_text = entry["completion"].lower()
        full_text = f"{prompt_text} {completion_text}"

        # Calculate matches in different fields with different weights
        prompt_matches = sum(1 for qw in query_words if qw in prompt_text)
        completion_matches = sum(1 for qw in query_words if qw in completion_text)
        full_text_matches = sum(1 for qw in query_words if qw in full_text)
        
        # Weighted scoring:
        total_score = (prompt_matches * 3) + (completion_matches * 2) + full_text_matches
        
        # Boost score if cancer type or genes match query
        if entry["cancer_type"] and any(qw in entry["cancer_type"].lower() for qw in query_words):
            total_score += 2
        if entry["genes"] and any(qw in entry["genes"].lower() for qw in query_words):
            total_score += 2
            
        matched_keywords = {qw for qw in query_words if qw in full_text}
        
        ranked_results.append({
            "entry": entry,
            "score": total_score,
            "prompt_matches": prompt_matches,
            "completion_matches": completion_matches,
            "matched_keywords": matched_keywords
        })

    # Sort by score, then by number of prompt matches
    ranked_results.sort(key=lambda x: (-x["score"], -x["prompt_matches"]))
    return ranked_results

def filter_results(results, min_score, keyword_filters, cancer_types, genes):
    filtered = []
    for result in results:
        if result["score"] < min_score:
            continue
            
        entry = result["entry"]
        matched = True
        
        # Keyword filters (must match ALL if specified)
        if keyword_filters:
            text = f"{entry['prompt']} {entry['completion']}".lower()
            matched &= all(kw.lower() in text for kw in keyword_filters)
        
        # Cancer type filters (OR logic if multiple selected)
        if cancer_types and entry["cancer_type"]:
            entry_types = {ct.strip().lower() for ct in entry["cancer_type"].split(",")}
            matched &= any(ct.lower() in entry_types for ct in cancer_types)
        
        # Gene filters (OR logic if multiple selected)
        if genes and entry["genes"]:
            entry_genes = {g.strip().lower() for g in entry["genes"].split(",")}
            matched &= any(g.lower() in entry_genes for g in genes)
        
        if matched:
            filtered.append(result)
    
    return filtered

def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("Search clinical trial information with natural language queries.")
    
    if st.session_state.suggestions:
        st.subheader("💡 Sample Questions")
        cols = st.columns(2)
        for i, suggestion in enumerate(st.session_state.suggestions[:6]):
            with cols[i % 2]:
                if st.button(suggestion[:50] + ("..." if len(suggestion) > 50 else ""), 
                           key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.session_state.show_home = False
                    st.rerun()

    with st.form("search_form"):
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query,
            placeholder="What is the response rate for atezolizumab in NSCLC?"
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

def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.rerun()

    st.title("🔍 Search Results")
    
    with st.sidebar:
        st.subheader("🔎 Filters")
        st.session_state.min_score = st.slider(
            "Minimum Match Score", 0, 20, 1,
            help="Lower value will show more results, higher value shows only best matches")
        
        st.markdown("**Keyword Filters**")
        if st.session_state.keyword_filters:
            st.write("Current filters:")
            cols = st.columns(3)
            for i, kw in enumerate(st.session_state.keyword_filters):
                with cols[i % 3]:
                    if st.button(f"❌ {kw}", key=f"remove_kw_{i}"):
                        st.session_state.keyword_filters.remove(kw)
                        st.rerun()
        
        new_keyword = st.text_input("Add keyword filter", key="new_keyword",
                                   help="Filter results to only show entries containing this word")
        if st.button("Add Filter") and new_keyword.strip():
            if new_keyword.lower() not in [k.lower() for k in st.session_state.keyword_filters]:
                st.session_state.keyword_filters.append(new_keyword.strip())
                st.rerun()
        
        if st.session_state.cancer_types:
            st.markdown("**Cancer Types**")
            selected = st.multiselect(
                "Select types", st.session_state.cancer_types,
                default=st.session_state.cancer_type_filter,
                help="Filter by cancer type (select multiple)")
            if selected != st.session_state.cancer_type_filter:
                st.session_state.cancer_type_filter = selected
                st.rerun()
        
        if st.session_state.genes:
            st.markdown("**Genes**")
            selected = st.multiselect(
                "Select genes", st.session_state.genes,
                default=st.session_state.gene_filter,
                help="Filter by gene (select multiple)")
            if selected != st.session_state.gene_filter:
                st.session_state.gene_filter = selected
                st.rerun()
        
        if st.button("Clear All Filters"):
            st.session_state.keyword_filters = []
            st.session_state.cancer_type_filter = []
            st.session_state.gene_filter = []
            st.session_state.min_score = 1
            st.rerun()

    data, word_index, _, _, _ = load_and_index_data()
    if data is None:
        return

    st.markdown(f"**Query:** {st.session_state.current_query}")

    with st.spinner("Searching..."):
        results = keyword_search(st.session_state.current_query, data, word_index)
        filtered = filter_results(
            results,
            st.session_state.min_score,
            st.session_state.keyword_filters,
            st.session_state.cancer_type_filter,
            st.session_state.gene_filter
        ) if results else []

        if filtered:
            display_results(filtered, results)
        else:
            show_no_results(data, word_index)

def display_results(results, all_results):
    st.success(f"Showing {len(results)} of {len(all_results)} matches")
    
    if len(results) > 1:
        st.bar_chart(pd.DataFrame({"Score": [r["score"] for r in results]}))
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download JSON",
            json.dumps([r["entry"] for r in results], indent=2),
            "results.json",
            "application/json"
        )
    with col2:
        st.download_button(
            "Download CSV",
            pd.DataFrame([r["entry"] for r in results]).to_csv(index=False),
            "results.csv",
            "text/csv"
        )

    for i, result in enumerate(results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']} - {entry['prompt'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Question:** {entry['prompt']}")
            st.markdown(f"**Answer:** {entry['completion']}")
            
            metadata = []
            if entry.get("nct_id"):
                metadata.append(f"**NCT ID:** {entry['nct_id']}")
            if entry["cancer_type"]:
                metadata.append(f"**Cancer Type:** {entry['cancer_type']}")
            if entry["genes"]:
                metadata.append(f"**Genes:** {entry['genes']}")
            if metadata:
                st.markdown(" | ".join(metadata))
            
            if entry["other_data"]:
                with st.expander("Additional Data"):
                    for k, v in entry["other_data"].items():
                        st.markdown(f"**{k.title()}:** {v}")

def show_no_results(data, word_index):
    st.error("No matches found. Try:")
    
    # Show spelling suggestions
    suggestions = get_search_suggestions(st.session_state.current_query, word_index)
    if suggestions:
        st.write("**Did you mean?**")
        corrected = " ".join(suggestions)
        if st.button(corrected):
            st.session_state.current_query = corrected
            st.rerun()
    
    # Show similar questions
    similar_questions = set()
    query_words = {w.lower() for w in st.session_state.current_query.split() if len(w) > 3}
    
    if query_words and word_index and data:
        for word in list(query_words)[:3]:
            if word in word_index:
                for doc_id in word_index[word][:3]:
                    if doc_id < len(data):
                        similar_questions.add(data[doc_id]["prompt"])
    
    if similar_questions:
        st.write("**Similar questions:**")
        cols = st.columns(2)
        for i, sugg in enumerate(similar_questions):
            with cols[i % 2]:
                if st.button(sugg[:50] + ("..." if len(sugg) > 50 else ""), key=f"sugg_{i}"):
                    st.session_state.current_query = sugg
                    st.rerun()
    
    st.markdown("""
    **Tips:**
    - Reduce minimum score
    - Remove filters
    - Check spelling
    - Use simpler terms
    - Try broader search terms
    """)

def main():
    st.set_page_config(
        page_title="Cancer Clinical Search",
        page_icon="🧬",
        layout="wide"
    )

    if not st.session_state.search_history:
        st.session_state.search_history = load_search_history()

    if not st.session_state.suggestions:
        _, _, cancer_types, genes, suggestions = load_and_index_data()
        st.session_state.suggestions = suggestions
        st.session_state.cancer_types = cancer_types
        st.session_state.genes = genes

    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Cancer+Search", width=150)
        st.markdown("""
        **About:**
        - Search clinical trial data
        - Filter by cancer type/genes
        - Export results
        """)
        
        if st.session_state.search_history:
            with st.expander("Search History", expanded=False):
                for i, item in enumerate(reversed(st.session_state.search_history[-5:])):
                    if st.button(item["query"][:50] + ("..." if len(item["query"]) > 50 else ""), 
                               key=f"history_{i}"):
                        st.session_state.current_query = item["query"]
                        st.session_state.show_home = False
                        st.rerun()

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
