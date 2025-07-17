import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import random
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap
import plotly.express as px

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_home' not in st.session_state:
    st.session_state.show_home = True
if 'min_score' not in st.session_state:
    st.session_state.min_score = 0.4
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
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Constants
DATA_FILE = "cancer_clinical_dataset.json"
HISTORY_FILE = "search_history.json"
MODEL_NAME = "all-MiniLM-L6-v2"

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

# Load and preprocess data with caching
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_and_process_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        clean_data = []
        cancer_types = set()
        genes = set()
        all_prompts = []
        all_texts = []

        for entry in raw_data:
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                # Clean and standardize data
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                text = f"{entry['prompt']} {entry['completion']}"
                
                # Extract metadata
                entry_cancer_types = []
                if "cancer_type" in entry:
                    entry_cancer_types = [ct.strip() for ct in str(entry["cancer_type"]).split(",")]
                    cancer_types.update(entry_cancer_types)
                
                entry_genes = []
                if "genes" in entry:
                    entry_genes = [g.strip() for g in str(entry["genes"]).split(",")]
                    genes.update(entry_genes)
                
                # Store cleaned entry
                clean_data.append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "cancer_type": ", ".join(entry_cancer_types) if entry_cancer_types else "",
                    "genes": ", ".join(entry_genes) if entry_genes else "",
                    "text": text
                })
                all_prompts.append(entry["prompt"])
                all_texts.append(text)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None, [], [], [], None, None
        
        # Generate embeddings
        model = load_model()
        embeddings = model.encode(all_texts, show_progress_bar=False)
        
        # Cluster similar entries
        n_clusters = min(10, len(clean_data))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            cluster_labels = np.zeros(len(clean_data))
        
        # Generate random suggestions
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        return clean_data, sorted(cancer_types), sorted(genes), random_suggestions, embeddings, cluster_labels
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

# Semantic search function
def semantic_search(query, dataset, embeddings, model, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_results = np.argsort(cos_scores)[-top_k:][::-1]
    return [(i, cos_scores[i]) for i in top_results]

# Find related concepts using TF-IDF
def find_related_concepts(dataset, query, top_n=5):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([entry["text"] for entry in dataset])
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_indices = similarities.argsort()[-top_n:][::-1]
    
    # Extract keywords from top matches
    feature_names = vectorizer.get_feature_names_out()
    keywords = set()
    for idx in related_indices:
        entry = dataset[idx]
        words = set(entry["text"].lower().split())
        keywords.update(w for w in words if len(w) > 3 and w in feature_names)
    
    return list(keywords)[:top_n]

# Visualize clusters
def visualize_clusters(embeddings, labels):
    reducer = umap.UMAP(random_state=42)
    umap_embeds = reducer.fit_transform(embeddings)
    
    fig = px.scatter(
        x=umap_embeds[:, 0],
        y=umap_embeds[:, 1],
        color=labels,
        title="Knowledge Cluster Visualization",
        labels={'color': 'Cluster'},
        width=800,
        height=600
    )
    st.plotly_chart(fig)

# Home page layout
def show_home():
    st.title("🧬 Cancer Clinical Search with Neural Network")
    st.markdown("""
    **Advanced search tool connecting cancer research concepts using neural networks**  
    Discover hidden relationships between treatments, genes, and outcomes.
    """)

    # Load data
    data, cancer_types, genes, suggestions, embeddings, cluster_labels = load_and_process_data()
    if data is None:
        return

    # Store in session state
    st.session_state.cancer_types = cancer_types
    st.session_state.genes = genes
    st.session_state.suggestions = suggestions
    st.session_state.embeddings = embeddings
    st.session_state.cluster_labels = cluster_labels
    st.session_state.model = load_model()

    # Display cluster visualization
    with st.expander("🔍 Knowledge Cluster Visualization", expanded=True):
        if embeddings is not None and cluster_labels is not None:
            visualize_clusters(embeddings, cluster_labels)
        else:
            st.info("Cluster visualization not available")

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

    # Search interface
    with st.form("search_form"):
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query,
            placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high NSCLC patients?",
            help="Enter your clinical question or keywords"
        )

        col1, col2 = st.columns(2)
        with col1:
            search_type = st.radio("Search type:", ["Semantic", "Keyword"])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.form_submit_button("Search", type="primary")

        if search_btn and query.strip():
            st.session_state.current_query = query
            st.session_state.show_home = False
            st.session_state.search_history.append({
                "query": query,
                "type": search_type,
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
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Higher scores mean better matches"
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
            st.session_state.min_score = 0.4
            st.rerun()

    # Get search type from history
    search_type = "Semantic"  # default
    if st.session_state.search_history:
        last_search = next((s for s in reversed(st.session_state.search_history) 
                          if s["query"] == st.session_state.current_query), None)
        if last_search and "type" in last_search:
            search_type = last_search["type"]

    st.markdown(f"**Current Search:** {st.session_state.current_query} ({search_type} search)")

    # Load data
    data, _, _, _, embeddings, _ = load_and_process_data()
    if data is None:
        return

    with st.spinner("Searching clinical knowledge base..."):
        if search_type == "Semantic" and st.session_state.model and embeddings is not None:
            # Semantic search
            model = st.session_state.model
            results = semantic_search(st.session_state.current_query, data, embeddings, model, top_k=50)
            ranked_results = []
            
            for idx, score in results:
                if score >= st.session_state.min_score:
                    ranked_results.append({
                        "entry": data[idx],
                        "score": score,
                        "matched_keywords": find_related_concepts(data, data[idx]["text"])
                    })
        else:
            # Keyword search fallback
            query_words = set(word.lower() for word in st.session_state.current_query.split() if len(word) > 2)
            ranked_results = []
            
            for idx, entry in enumerate(data):
                prompt_words = set(entry["prompt"].lower().split())
                completion_words = set(entry["completion"].lower().split())
                
                prompt_matches = len(query_words & prompt_words)
                completion_matches = len(query_words & completion_words)
                total_score = (prompt_matches + completion_matches) / max(1, len(query_words))
                
                if total_score >= st.session_state.min_score:
                    ranked_results.append({
                        "entry": entry,
                        "score": total_score,
                        "matched_keywords": query_words & (prompt_words | completion_words)
                    })
            
            ranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply filters
        filtered_results = []
        for result in ranked_results:
            entry = result["entry"]
            
            # Apply keyword filters
            if st.session_state.keyword_filters:
                matched = any(kw.lower() in result.get("matched_keywords", set()) 
                            for kw in st.session_state.keyword_filters)
                if not matched:
                    continue
            
            # Apply cancer type filter
            if st.session_state.cancer_type_filter and entry["cancer_type"]:
                entry_types = set(ct.strip() for ct in entry["cancer_type"].split(","))
                if not any(ct in entry_types for ct in st.session_state.cancer_type_filter):
                    continue
            
            # Apply gene filter
            if st.session_state.gene_filter and entry["genes"]:
                entry_genes = set(g.strip() for g in entry["genes"].split(","))
                if not any(g in entry_genes for g in st.session_state.gene_filter):
                    continue
            
            filtered_results.append(result)

        if filtered_results:
            display_results(filtered_results, search_type)
        else:
            show_no_results(data)

def display_results(results, search_type):
    st.success(f"Found {len(results)} relevant results (using {search_type} search)")
    
    # Score distribution chart
    if len(results) > 1:
        scores = [r["score"] for r in results]
        st.bar_chart(pd.DataFrame({"Score": scores}), use_container_width=True)
    
    # Related concepts from top results
    if search_type == "Semantic":
        with st.expander("🔗 Related Concepts from Top Results", expanded=True):
            all_keywords = set()
            for result in results[:5]:
                all_keywords.update(result.get("matched_keywords", []))
            
            if all_keywords:
                cols = st.columns(4)
                for i, kw in enumerate(list(all_keywords)[:8]):
                    with cols[i % 4]:
                        if st.button(kw, key=f"related_kw_{i}"):
                            st.session_state.current_query = kw
                            st.rerun()
    
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
        with st.expander(f"#{i} | Score: {result['score']:.2f} - {entry['prompt'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Question:** {entry['prompt']}")
            st.markdown(f"**Answer:** {entry['completion']}")
            
            # Metadata
            metadata = []
            if entry["cancer_type"]:
                metadata.append(f"**Cancer Type:** {entry['cancer_type']}")
            if entry["genes"]:
                metadata.append(f"**Genes:** {entry['genes']}")
            if metadata:
                st.markdown(" | ".join(metadata))
            
            # Related keywords
            if "matched_keywords" in result and result["matched_keywords"]:
                st.markdown(f"**Related Keywords:** {', '.join(result['matched_keywords'])}")
            
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

def show_no_results(data):
    st.error("No matches found with current filters. Try these suggestions:")
    
    # Generate suggestions from similar entries
    if st.session_state.model and st.session_state.embeddings is not None:
        similar_results = semantic_search(
            st.session_state.current_query, 
            data, 
            st.session_state.embeddings, 
            st.session_state.model,
            top_k=5
        )
        
        if similar_results:
            st.write("**Semantically similar questions:**")
            cols = st.columns(2)
            for i, (idx, score) in enumerate(similar_results):
                with cols[i % 2]:
                    if st.button(data[idx]["prompt"][:50] + "..." if len(data[idx]["prompt"]) > 50 else data[idx]["prompt"], 
                                key=f"similar_{i}"):
                        st.session_state.current_query = data[idx]["prompt"]
                        st.rerun()
    
    st.markdown("""
    **Search Tips:**
    - Try lowering the minimum score filter
    - Remove some filters to broaden your search
    - Check for typos in your search terms
    - Use more general terms if your search is too specific
    - Try switching between semantic and keyword search
    """)

# Main app flow
def main():
    st.set_page_config(
        page_title="🧬 Neural Cancer Clinical Search",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load search history
    if not st.session_state.search_history:
        st.session_state.search_history = load_search_history()

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
