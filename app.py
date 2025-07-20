# app.py

import streamlit as st
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import os
import random
import re

DATA_FILE = "cleaned_cancer_data.json"
HISTORY_FILE = "search_history.json"

CANCER_KEYWORD_MAP = {
    "lung cancer": ["nsclc", "sclc", "egfr", "alk"],
    "breast cancer": ["her2", "esr1", "triple negative", "brca1", "brca2"],
    "colorectal cancer": ["crc", "colon", "braf", "kras", "msi"],
    "prostate cancer": ["androgen", "psa", "enzalutamide"]
}

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

def expand_keywords(query, keyword_map):
    query_words = {w.lower() for w in query.split()}
    expanded = set(query_words)
    for word in query_words:
        for key, related in keyword_map.items():
            if word in related or word == key:
                expanded.update([key] + related)
    return expanded

@st.cache_data
def load_and_index_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Data file '{DATA_FILE}' not found.")
        st.stop()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")
            return [], {}, [], [], []

    clean_data = []
    word_index = defaultdict(list)
    cancer_types = set()
    genes = set()
    all_prompts = []

    for idx, entry in enumerate(raw_data):
        prompt = str(entry.get("prompt", "")).strip()
        completion = str(entry.get("completion", "")).strip()
        if not prompt and not completion:
            continue

        cancer_type = entry.get("cancer_type", "Unknown").strip().title()
        gene_str = entry.get("genes", "").strip()

        clean_entry = {
            "prompt": prompt,
            "completion": completion,
            "cancer_type": cancer_type,
            "genes": gene_str.upper(),
            "nct_id": entry.get("nct_id", ""),
            "other_data": entry.get("other_data", {})
        }

        for text in [prompt, completion]:
            for word in set(text.lower().split()):
                if len(word) > 2:
                    word_index[word].append(len(clean_data))

        clean_data.append(clean_entry)
        all_prompts.append(prompt)
        cancer_types.add(cancer_type)
        for g in gene_str.split(","):
            genes.add(g.strip().upper())

    return clean_data, word_index, sorted(cancer_types), sorted(genes), random.sample(all_prompts, min(10, len(all_prompts)))

def keyword_search(query, dataset, word_index):
    if not query or not dataset:
        return []
    query_words = expand_keywords(query, CANCER_KEYWORD_MAP)
    doc_matches = defaultdict(int)

    for word in query_words:
        if word in word_index:
            for doc_id in word_index[word]:
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
            matched &= any(g.strip().upper() in entry["genes"] for g in genes)

        if matched:
            filtered.append(result)
    return filtered

def save_search_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.search_history, f, indent=2)

def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("Search cancer trial prompt-completion data.")
    if st.session_state.suggestions:
        st.subheader("💡 Sample Prompts")
        cols = st.columns(2)
        for i, sugg in enumerate(st.session_state.suggestions[:6]):
            with cols[i % 2]:
                if st.button(sugg[:50], key=f"sugg_{i}"):
                    st.session_state.current_query = sugg
                    st.session_state.show_home = False
                    st.rerun()

    with st.form("search"):
        query = st.text_input("Ask a question:", st.session_state.current_query)
        if st.form_submit_button("Search"):
            st.session_state.current_query = query
            st.session_state.search_history.append({"query": query, "timestamp": datetime.now().isoformat()})
            save_search_history()
            st.session_state.show_home = False
            st.rerun()

def show_results():
    if st.button("← Back"):
        st.session_state.show_home = True
        st.rerun()

    st.title("🔍 Search Results")
    with st.sidebar:
        st.subheader("🔧 Filters")
        st.session_state.min_score = st.slider("Min Score", 0, 20, 1)
        new_kw = st.text_input("Keyword")
        if st.button("Add Keyword") and new_kw:
            st.session_state.keyword_filters.append(new_kw)
            st.rerun()
        st.multiselect("Cancer Type", st.session_state.cancer_types, default=st.session_state.cancer_type_filter, key="cancer_type_filter")
        st.multiselect("Genes", st.session_state.genes, default=st.session_state.gene_filter, key="gene_filter")
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
        st.success(f"Showing {len(filtered)} matches")
        for i, result in enumerate(filtered, 1):
            with st.expander(f"#{i} | Score: {result['score']}"):
                entry = result["entry"]
                st.markdown(f"**Prompt:** {entry['prompt']}")
                st.markdown(f"**Answer:** {entry['completion']}")
                st.markdown(f"NCT ID: {entry.get('nct_id', '-')}")
                st.markdown(f"Cancer Type: {entry['cancer_type']} | Genes: {entry['genes']}")
    else:
        st.warning("No matches. Try relaxing filters or using simpler keywords.")

def main():
    st.set_page_config("Cancer Search", page_icon="🧬", layout="wide")
    if not st.session_state.suggestions:
        _, _, c_types, genes, suggs = load_and_index_data()
        st.session_state.cancer_types = c_types
        st.session_state.genes = genes
        st.session_state.suggestions = suggs

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
