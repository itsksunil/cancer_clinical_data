import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os

# ---------------------------
# Hierarchical keyword map
# ---------------------------
CANCER_KEYWORD_MAP = {
    "lung cancer": ["nsclc", "egfr", "alk", "sclc", "pd-l1"],
    "breast cancer": ["her2", "er+", "esr1", "triple negative", "brca1", "brca2"],
    "colorectal cancer": ["colon", "crc", "braf", "msi", "ras"],
    "prostate cancer": ["androgen", "psa", "ar signaling", "enzalutamide"]
}

# ---------------------------
# Session initialization
# ---------------------------
for key, default in {
    'search_history': [],
    'current_query': "",
    'show_home': True,
    'min_score': 1,
    'keyword_filters': [],
    'cancer_type_filter': [],
    'gene_filter': [],
    'suggestions': [],
    'cancer_types': [],
    'genes': [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

DATA_FILE = "cancer_clinical_dataset.json"
HISTORY_FILE = "search_history.json"

# ---------------------------
# Utility Functions
# ---------------------------
def expand_keywords(query, keyword_map):
    query_words = {w.lower() for w in query.split()}
    expanded = set(query_words)
    for word in query_words:
        for key, related in keyword_map.items():
            if word in related or word == key:
                expanded.update([key] + related)
    return expanded

def validate_data_file():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file '{DATA_FILE}' not found.")
        st.stop()

@st.cache_data
def load_and_index_data():
    validate_data_file()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    clean_data = []
    word_index = defaultdict(list)
    cancer_types = set()
    genes = set()
    all_prompts = []

    for idx, entry in enumerate(raw_data):
        prompt = entry.get("prompt", "").strip()
        completion = entry.get("completion", "").strip()
        if not prompt and not completion:
            continue

        cancer_type = entry.get("cancer_type", "").strip().lower()
        gene_str = entry.get("genes", "").strip()

        clean_entry = {
            "prompt": prompt,
            "completion": completion,
            "cancer_type": cancer_type.title(),
            "genes": gene_str.upper(),
            "nct_id": entry.get("nct_id", ""),
            "other_data": {k: v for k, v in entry.items() if k not in ["prompt", "completion", "cancer_type", "genes", "nct_id"]}
        }

        for text in [prompt, completion]:
            for word in set(text.lower().split()):
                if len(word) > 2 and word.isalpha():
                    word_index[word].append(idx)

        clean_data.append(clean_entry)
        all_prompts.append(prompt)

        if cancer_type:
            cancer_types.add(cancer_type.title())
        if gene_str:
            for g in gene_str.split(","):
                genes.add(g.strip().upper())

    random_suggestions = random.sample(all_prompts, min(10, len(all_prompts)))
    return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions

def keyword_search(query, dataset, word_index):
    if not query or not dataset:
        return []
    query_words = expand_keywords(query, CANCER_KEYWORD_MAP)
    doc_matches = defaultdict(int)

    for word in query_words:
        if word in word_index:
            for doc_id in word_index[word]:
                if doc_id < len(dataset):
                    doc_matches[doc_id] += 1

    results = []
    for doc_id, count in doc_matches.items():
        entry = dataset[doc_id]
        prompt_words = set(entry["prompt"].lower().split())
        completion_words = set(entry["completion"].lower().split())
        match_words = query_words & (prompt_words | completion_words)
        score = len(query_words & prompt_words) * 2 + len(query_words & completion_words)
        results.append({"entry": entry, "score": score, "matched_keywords": match_words})

    return sorted(results, key=lambda x: x["score"], reverse=True)

def filter_results(results, min_score, keyword_filters, cancer_types, genes):
    filtered = []
    for result in results:
        entry = result["entry"]
        if result["score"] < min_score:
            continue

        matched = True
        if keyword_filters:
            matched &= any(kw.lower() in result["matched_keywords"] for kw in keyword_filters)
        if cancer_types and entry["cancer_type"]:
            matched &= entry["cancer_type"] in cancer_types
        if genes and entry["genes"]:
            entry_genes = {g.strip() for g in entry["genes"].split(",")}
            matched &= any(g in entry_genes for g in genes)

        if matched:
            filtered.append(result)
    return filtered

def save_search_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.search_history, f, indent=2)

# ---------------------------
# UI Components
# ---------------------------
def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("Use natural queries to search cancer trial insights.")

    if st.session_state.suggestions:
        st.subheader("💡 Try these:")
        cols = st.columns(2)
        for i, sugg in enumerate(st.session_state.suggestions[:6]):
            with cols[i % 2]:
                if st.button(sugg[:50], key=f"sugg_{i}"):
                    st.session_state.current_query = sugg
                    st.session_state.show_home = False
                    st.rerun()

    with st.form("search"):
        query = st.text_input("Ask your question:", st.session_state.current_query)
        if st.form_submit_button("Search"):
            st.session_state.current_query = query
            st.session_state.search_history.append({"query": query, "timestamp": datetime.now().isoformat()})
            save_search_history()
            st.session_state.show_home = False
            st.rerun()

def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.rerun()

    st.title("🔍 Search Results")
    with st.sidebar:
        st.subheader("Filters")
        st.session_state.min_score = st.slider("Min Score", 0, 20, 1)
        st.markdown("**Add Keywords**")
        new_kw = st.text_input("Keyword")
        if st.button("Add Keyword") and new_kw:
            if new_kw not in st.session_state.keyword_filters:
                st.session_state.keyword_filters.append(new_kw)
                st.rerun()
        st.markdown("**Cancer Types**")
        selected_types = st.multiselect("Types", st.session_state.cancer_types, default=st.session_state.cancer_type_filter)
        st.session_state.cancer_type_filter = selected_types
        st.markdown("**Genes**")
        selected_genes = st.multiselect("Genes", st.session_state.genes, default=st.session_state.gene_filter)
        st.session_state.gene_filter = selected_genes
        if st.button("Clear Filters"):
            st.session_state.keyword_filters.clear()
            st.session_state.cancer_type_filter.clear()
            st.session_state.gene_filter.clear()
            st.session_state.min_score = 1
            st.rerun()

    data, word_index, _, _, _ = load_and_index_data()
    results = keyword_search(st.session_state.current_query, data, word_index)
    filtered = filter_results(results, st.session_state.min_score, st.session_state.keyword_filters,
                              st.session_state.cancer_type_filter, st.session_state.gene_filter)

    st.markdown(f"**Query:** {st.session_state.current_query}")
    if filtered:
        st.success(f"{len(filtered)} matches found.")
        st.download_button("Download JSON", json.dumps([r["entry"] for r in filtered], indent=2), "results.json")
        for i, result in enumerate(filtered, 1):
            with st.expander(f"#{i} | Score: {result['score']}"):
                entry = result["entry"]
                st.markdown(f"**Q:** {entry['prompt']}")
                st.markdown(f"**A:** {entry['completion']}")
                if entry["nct_id"]:
                    st.markdown(f"**NCT ID:** {entry['nct_id']}")
                if entry["cancer_type"]:
                    st.markdown(f"**Cancer Type:** {entry['cancer_type']}")
                if entry["genes"]:
                    st.markdown(f"**Genes:** {entry['genes']}")
                if entry["other_data"]:
                    st.json(entry["other_data"])
    else:
        st.warning("No results matched. Try simplifying your query or adjusting filters.")

# ---------------------------
# Main App Entry
# ---------------------------
def main():
    st.set_page_config("Cancer Clinical Search", page_icon="🧬", layout="wide")
    if not st.session_state.suggestions:
        _, _, c_types, g_list, suggs = load_and_index_data()
        st.session_state.cancer_types = c_types
        st.session_state.genes = g_list
        st.session_state.suggestions = suggs

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
