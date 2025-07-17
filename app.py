import streamlit as st
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from collections import defaultdict

# --- Constants ---
COMMON_KEYWORDS = [
    "side effects", "surgery", "cancer type", "how", "what", "when",
    "survival rate", "dosage", "mechanism", "biomarkers", "efficacy",
    "adverse events", "progression", "response rate", "combination therapy"
]

# --- Load Embedding Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Load and Clean JSON Dataset ---
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    processed_data = []
    for d in raw_data:
        if isinstance(d, dict) and "prompt" in d and "completion" in d:
            # Extract drug names
            drugs = []
            for drug in ["nivolumab", "pembrolizumab", "atezolizumab"]:
                if drug in d["prompt"].lower() or drug in d["completion"].lower():
                    drugs.append(drug.capitalize())
            
            d["drugs"] = drugs
            d["word_count"] = len(d["completion"].split())
            processed_data.append(d)
    
    return processed_data

# --- Compute Embeddings ---
@st.cache_data
def compute_prompt_embeddings(prompts):
    model = load_model()
    return model.encode(prompts, show_progress_bar=False)

# --- Query Logger ---
def log_query(query, top_prompt):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "top_prompt": top_prompt
    }
    with open("query_log.csv", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_data) + "\n")

# --- Streamlit UI ---
st.set_page_config(page_title="🧬 Neural Search for Cancer Trials", layout="wide")
st.title("🔍 LLM-Enhanced Cancer Clinical Q&A Explorer")
st.markdown("Search structured clinical answers using deep neural embeddings")

# Sidebar with quick access
with st.sidebar:
    st.header("Quick Access Keywords")
    cols = st.columns(3)
    for i, keyword in enumerate(COMMON_KEYWORDS):
        with cols[i % 3]:
            if st.button(keyword.title()):
                st.session_state.query = keyword
    
    st.header("Analysis Tools")
    show_stats = st.checkbox("Show Dataset Statistics")

# Load data
data = load_data()
prompts = [item["prompt"] for item in data]
model = load_model()
prompt_embeddings = compute_prompt_embeddings(prompts)

# Main search interface
query = st.text_input("Ask your clinical question:", value=st.session_state.get("query", ""))

if query:
    # Perform search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    top_results = [data[i] for i in ranked_indices[:5]]
    
    # Display results
    st.markdown("### 🔎 Top 5 Matches")
    for rank, idx in enumerate(ranked_indices[:5], start=1):
        score = similarities[idx]
        entry = data[idx]
        
        color = "green" if score > 0.7 else "blue" if score > 0.5 else "orange"
        
        st.markdown(f"""
        <div style="border-left: 5px solid {color}; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4>Rank {rank} | Similarity Score: {score:.2f}</h4>
            <p><strong>Prompt:</strong> {entry['prompt']}</p>
            <p><strong>Answer:</strong> {entry['completion']}</p>
            {f"<p><strong>Drugs:</strong> {', '.join(entry['drugs'])}</p>" if entry['drugs'] else ""}
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📄 Detailed View"):
            st.json(entry)
        st.markdown("---")
    
    # Visualizations using Streamlit native
    st.subheader("📈 Results Analysis")
    
    # Create DataFrame for visualization
    viz_data = {
        'Prompt': [d['prompt'][:30]+'...' for d in top_results],
        'Similarity': [similarities[i] for i in ranked_indices[:5]],
        'Drug Count': [len(d['drugs']) for d in top_results],
        'Answer Length': [d['word_count'] for d in top_results]
    }
    df = pd.DataFrame(viz_data)
    
    # Bar chart
    st.bar_chart(df.set_index('Prompt')['Similarity'])
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest Similarity", f"{max(df['Similarity']):.2f}")
    with col2:
        st.metric("Avg Drug Count", f"{df['Drug Count'].mean():.1f}")
    with col3:
        st.metric("Avg Answer Length", f"{df['Answer Length'].mean():.1f} words")
    
    # Data table
    st.dataframe(df)
    
    # Log and download
    log_query(query, top_results[0]["prompt"])
    st.download_button("⬇️ Download Results as CSV", 
                      df.to_csv(index=False), 
                      "clinical_search_results.csv", 
                      "text/csv")

# Dataset statistics
if show_stats:
    st.subheader("📊 Dataset Statistics")
    
    # Drug frequency
    drug_counts = defaultdict(int)
    for entry in data:
        for drug in entry.get("drugs", []):
            drug_counts[drug] += 1
    
    if drug_counts:
        st.write("**Drug Frequency:**")
        drug_df = pd.DataFrame.from_dict(drug_counts, orient="index", columns=["Count"])
        st.dataframe(drug_df.sort_values("Count", ascending=False))
    
    # Word count stats
    word_counts = [entry["word_count"] for entry in data]
    word_df = pd.DataFrame(word_counts, columns=["Word Count"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Answer Length Distribution**")
        st.bar_chart(word_df.value_counts())
    with col2:
        st.write("**Summary Statistics**")
        st.table(word_df.describe())

else:
    st.info("💡 Enter a clinical question to begin semantic search. Try the quick access keywords in the sidebar!")
