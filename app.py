import streamlit as st
import json
from difflib import get_close_matches

# Load the JSON dataset
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

# App title and description
st.set_page_config(page_title="Cancer Clinical Q&A", layout="centered")
st.title("🧬 Cancer Clinical Trial Q&A")
st.markdown("Ask any question about cancer trials, biomarkers, immune response, or clinical outcomes.")

# User input
user_question = st.text_input("🔍 Ask your clinical question here:")

# Search logic: Fuzzy matching + keyword fallback
def find_top_matches(question, dataset, top_n=3):
    question_lower = question.lower()
    prompts = [entry["prompt"] for entry in dataset]

    # Fuzzy match
    close_matches = get_close_matches(question, prompts, n=top_n, cutoff=0.4)
    matched_results = [entry for entry in dataset if entry["prompt"] in close_matches]

    # Fallback: keyword match
    if not matched_results:
        for entry in dataset:
            prompt = entry["prompt"].lower()
            if any(word in prompt for word in question_lower.split()):
                matched_results.append(entry)

    return matched_results[:top_n]

# Display results
if user_question:
    results = find_top_matches(user_question, data)
    if results:
        for i, result in enumerate(results, 1):
            st.success(f"✅ Match {i}")
            st.markdown(f"**Prompt:** {result['prompt']}")
            st.markdown(f"**Answer:** {result['completion']}")
            st.markdown("---")
    else:
        st.error("❌ No relevant match found. Try rephrasing or using different keywords.")
