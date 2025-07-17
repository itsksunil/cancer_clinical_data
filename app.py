import streamlit as st
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import plotly.express as px
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
    
    # Process data to extract additional metadata
    processed_data = []
    for d in raw_data:
        if isinstance(d, dict) and "prompt" in d and "completion" in d:
            # Extract drug names (simple pattern matching)
            drugs = []
            if "nivolumab" in d["prompt"].lower() or "nivolumab" in d["completion"].lower():
                drugs.append("Nivolumab")
            if "pembrolizumab" in d["prompt"].lower() or "pembrolizumab" in d["completion"].lower():
                drugs.append("Pembrolizumab")
            if "atezolizumab" in d["prompt"].lower() or "atezolizumab" in d["completion"].lower():
                drugs.append("Atezolizumab")
            
            # Add metadata
            d["drugs"] = drugs
            d["word_count"] = len(d["completion"].split())
            processed_data.append(d)
    
    return processed_data

# --- Compute Embeddings for Prompts ---
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

# --- Generate Drug Network Graph ---
def generate_drug_network(data):
    drug_connections = defaultdict(int)
    drug_counts = defaultdict(int)
    
    for entry in data:
        drugs = entry.get("drugs", [])
        for i in range(len(drugs)):
            drug_counts[drugs[i]] += 1
            for j in range(i+1, len(drugs)):
                pair = tuple(sorted((drugs[i], drugs[j])))
                drug_connections[pair] += 1
    
    # Prepare data for visualization
    sources = []
    targets = []
    values = []
    drug_sizes = []
    
    for pair, count in drug_connections.items():
        sources.append(pair[0])
        targets.append(pair[1])
        values.append(count)
    
    for drug, count in drug_counts.items():
        sources.append(drug)
        targets.append(drug)  # Self-reference for size
        values.append(count)
        drug_sizes.append(count)
    
    return sources, targets, values, drug_sizes

# --- Streamlit UI ---
st.set_page_config(page_title="🧬 Neural Search for Cancer Trials", layout="wide")
st.title("🔍 LLM-Enhanced Cancer Clinical Q&A Explorer")
st.markdown("Search structured clinical answers using deep neural embeddings (no fuzzy match, no GPT required).")

# Sidebar with quick access buttons
with st.sidebar:
    st.header("Quick Access Keywords")
    cols = st.columns(3)
    for i, keyword in enumerate(COMMON_KEYWORDS):
        with cols[i % 3]:
            if st.button(keyword.title()):
                st.session_state.query = keyword
    
    st.header("Drug Discovery Tools")
    show_network = st.checkbox("Show Drug Interaction Network")
    show_stats = st.checkbox("Show Dataset Statistics")

# Load and prep
data = load_data()
prompts = [item["prompt"] for item in data]
model = load_model()
prompt_embeddings = compute_prompt_embeddings(prompts)

# User Query
query = st.text_input("Ask your clinical question (e.g. 'What is OS in IMpower010?'):", 
                     value=st.session_state.get("query", ""))

if show_network:
    st.subheader("🧪 Drug Interaction Network")
    sources, targets, values, drug_sizes = generate_drug_network(data)
    
    if sources:
        fig = px.scatter(
            x=range(len(drug_sizes)),
            y=[1]*len(drug_sizes),
            size=drug_sizes,
            size_max=30,
            color=drug_sizes,
            hover_name=list(set(sources + targets)),
            labels={"x": "", "y": ""},
            title="Drug Co-occurrence in Clinical Trials (Size = Frequency)"
        )
        fig.update_layout(showlegend=False, xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No drug interaction data found in the dataset.")

if show_stats:
    st.subheader("📊 Dataset Statistics")
    
    # Count drugs
    drug_counts = defaultdict(int)
    for entry in data:
        for drug in entry.get("drugs", []):
            drug_counts[drug] += 1
    
    if drug_counts:
        st.write("**Drug Frequency in Trials:**")
        drug_df = pd.DataFrame.from_dict(drug_counts, orient="index", columns=["Count"])
        st.dataframe(drug_df.sort_values("Count", ascending=False))
    
    # Word count distribution
    word_counts = [entry["word_count"] for entry in data]
    fig = px.histogram(word_counts, nbins=20, title="Answer Length Distribution")
    st.plotly_chart(fig, use_container_width=True)

if query:
    # Embed and search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, prompt_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    top_results = [data[i] for i in ranked_indices[:5]]

    st.markdown("### 🔎 Top 5 Matches")
    for rank, idx in enumerate(ranked_indices[:5], start=1):
        score = similarities[idx]
        entry = data[idx]
        
        # Create a colored box based on similarity score
        if score > 0.7:
            color = "green"
        elif score > 0.5:
            color = "blue"
        else:
            color = "orange"
        
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

    # Log first result
    log_query(query, top_results[0]["prompt"])

    # Visualize results
    st.subheader("📈 Results Visualization")
    results_df = pd.DataFrame({
        "Prompt": [d["prompt"][:50] + "..." for d in top_results],
        "Similarity": [similarities[i] for i in ranked_indices[:5]],
        "Drug Count": [len(d["drugs"]) for d in top_results],
        "Answer Length": [d["word_count"] for d in top_results]
    })
    
    fig = px.bar(results_df, x="Prompt", y="Similarity", color="Drug Count",
                hover_data=["Answer Length"], title="Top Results Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # Downloads
    df = pd.DataFrame(top_results)
    json_str = json.dumps(top_results, indent=2)
    csv_str = df.to_csv(index=False)

    st.markdown("### 📁 Download Your Results")
    st.download_button("⬇️ Download JSON", data=json_str, file_name="top_results.json", mime="application/json")
    st.download_button("⬇️ Download CSV", data=csv_str, file_name="top_results.csv", mime="text/csv")
else:
    st.info("Enter a clinical question to begin semantic search. Try quick access keywords from the sidebar!")
