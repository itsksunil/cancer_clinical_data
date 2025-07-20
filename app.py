import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import os
import requests
from urllib.parse import urlparse
from fuzzywuzzy import fuzz # For fuzzy matching

# --- Session State Initialization ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_home' not in st.session_state:
    st.session_state.show_home = True
if 'show_settings' not in st.session_state: # New state for settings page
    st.session_state.show_settings = False
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
if 'remote_data_url' not in st.session_state:
    st.session_state.remote_data_url = ""
if 'use_remote_data' not in st.session_state:
    st.session_state.use_remote_data = False
if 'llm_api_key' not in st.session_state: # New: API key for LLM
    st.session_state.llm_api_key = ""
if 'llm_suggested_cancer_types' not in st.session_state: # New: LLM suggestions
    st.session_state.llm_suggested_cancer_types = []
if 'llm_suggested_genes' not in st.session_state: # New: LLM suggestions
    st.session_state.llm_suggested_genes = []
if 'ai_synthesized_answer' not in st.session_state: # New: LLM synthesized answer
    st.session_state.ai_synthesized_answer = ""

# --- Constants ---
DATA_FILES = ["cancer_clinical_dataset.json", "NCT02394626.json"] # Example local files
HISTORY_FILE = "search_history.json"
REMOTE_DATA_TIMEOUT = 10 # seconds
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"

# --- Utility Functions ---

def load_search_history():
    """Loads search history from a JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return [] # Return empty list on error
    return []

def save_search_history():
    """Saves search history to a JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f)
    except Exception:
        pass # Fail silently if cannot save history

def is_valid_url(url):
    """Checks if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def fetch_remote_data(url):
    """Fetches JSON data from a remote URL."""
    if not is_valid_url(url):
        st.error("Invalid URL provided.")
        return None
    
    try:
        response = requests.get(url, timeout=REMOTE_DATA_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Check if response is JSON
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            st.error("Remote data is not in JSON format. Please provide a URL to a JSON file.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching remote data: {str(e)}. Please check the URL or your network connection.")
        return None

def load_data_source():
    """Loads data from either a remote URL or multiple local JSON files."""
    if st.session_state.use_remote_data and st.session_state.remote_data_url:
        return fetch_remote_data(st.session_state.remote_data_url)
    else:
        all_local_data = []
        for file_name in DATA_FILES:
            if os.path.exists(file_name):
                try:
                    with open(file_name, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_local_data.extend(data)
                        else:
                            st.warning(f"Local file '{file_name}' does not contain a list of records. Skipping.")
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON from '{file_name}'. Please check its format.")
                except Exception as e:
                    st.error(f"Error loading local data from '{file_name}': {str(e)}")
            else:
                st.warning(f"Local data file not found: '{file_name}'. Ensure it's in the correct directory.")
        
        if not all_local_data:
            st.error("No local data found or loaded successfully. Please check your local files or switch to remote data.")
            return None
        return all_local_data

@st.cache_data
def load_and_index_data():
    """Loads and preprocesses data, creating a word index for search."""
    raw_data = load_data_source()
    if raw_data is None:
        return None, None, [], [], []
        
    clean_data = []
    word_index = defaultdict(list)
    cancer_types = set()
    genes = set()
    all_prompts = []

    for idx, entry in enumerate(raw_data):
        if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
            # Clean and standardize data
            prompt_text = str(entry["prompt"]).strip()
            completion_text = str(entry["completion"]).strip()
            
            # Extract metadata
            entry_cancer_types = [ct.strip() for ct in str(entry.get("cancer_type", "")).split(",") if ct.strip()]
            cancer_types.update(entry_cancer_types)
            
            entry_genes = [g.strip() for g in str(entry.get("genes", "")).split(",") if g.strip()]
            genes.update(entry_genes)
            
            # Store cleaned entry
            clean_data.append({
                "prompt": prompt_text,
                "completion": completion_text,
                "cancer_type": ", ".join(entry_cancer_types),
                "genes": ", ".join(entry_genes)
            })
            all_prompts.append(prompt_text)

            # Index words for exact matching
            for word in set(prompt_text.lower().split() + completion_text.lower().split()):
                if len(word) > 2: # Index words longer than 2 characters
                    word_index[word].append(idx)

    if not clean_data:
        st.error("No valid Q&A pairs found in the dataset(s) after processing.")
        return None, None, [], [], []
        
    random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
    return clean_data, word_index, sorted(list(cancer_types)), sorted(list(genes)), random_suggestions

def keyword_search(query, dataset, word_index):
    """Performs keyword search with fuzzy matching and ranks results."""
    if not query or not dataset:
        return []
        
    query_lower = query.lower()
    query_words = set(word for word in query_lower.split() if len(word) > 2)
    
    doc_scores = defaultdict(lambda: {'exact_matches': 0, 'fuzzy_prompt_score': 0, 'fuzzy_completion_score': 0})
    
    # Initial exact word matching to get a base set of documents
    potential_doc_ids = set()
    for word in query_words:
        if word in word_index:
            potential_doc_ids.update(word_index[word])
    
    # If no exact matches, consider all documents for fuzzy search (can be slow for very large datasets)
    if not potential_doc_ids and query_words:
        potential_doc_ids = set(range(len(dataset)))

    for doc_id in potential_doc_ids:
        if doc_id >= len(dataset): # Safety check
            continue

        entry = dataset[doc_id]
        
        # Calculate exact matches
        prompt_words_entry = set(entry["prompt"].lower().split())
        completion_words_entry = set(entry["completion"].lower().split())
        
        exact_prompt_matches = len(query_words.intersection(prompt_words_entry))
        exact_completion_matches = len(query_words.intersection(completion_words_entry))
        
        doc_scores[doc_id]['exact_matches'] = (exact_prompt_matches * 2) + exact_completion_matches # Prompt matches weighted higher

        # Calculate fuzzy scores
        doc_scores[doc_id]['fuzzy_prompt_score'] = fuzz.partial_ratio(query_lower, entry["prompt"].lower())
        doc_scores[doc_id]['fuzzy_completion_score'] = fuzz.partial_ratio(query_lower, entry["completion"].lower())
        
    ranked_results = []
    for doc_id, scores in doc_scores.items():
        entry = dataset[doc_id]
        
        # Combine scores: exact matches are most important, then fuzzy prompt, then fuzzy completion
        total_score = (scores['exact_matches'] * 100) + \
                      (scores['fuzzy_prompt_score'] * 0.5) + \
                      (scores['fuzzy_completion_score'] * 0.2)
        
        # Identify all matched keywords (exact or fuzzy) for display
        matched_keywords = set()
        for q_word in query_words:
            for p_word in entry["prompt"].lower().split():
                if fuzz.ratio(q_word, p_word) > 80: # High threshold for "matched keyword" display
                    matched_keywords.add(p_word)
            for c_word in entry["completion"].lower().split():
                if fuzz.ratio(q_word, c_word) > 80:
                    matched_keywords.add(c_word)

        ranked_results.append({
            "entry": entry,
            "score": total_score,
            "exact_prompt_matches": exact_prompt_matches, # For detailed display
            "exact_completion_matches": exact_completion_matches, # For detailed display
            "prompt_fuzzy_score": scores['fuzzy_prompt_score'],
            "completion_fuzzy_score": scores['fuzzy_completion_score'],
            "matched_keywords": matched_keywords
        })

    # Sort by total_score in descending order
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results

def filter_results(results, min_score, keyword_filters, cancer_types, genes):
    """Filters search results based on score, keywords, cancer types, and genes."""
    filtered = []
    for result in results:
        entry = result["entry"]
        
        # Apply score filter
        if result["score"] < min_score:
            continue
            
        # Apply keyword filters (logical OR for multiple keywords)
        if keyword_filters:
            matched_by_keyword_filter = False
            for kw_filter in keyword_filters:
                # Check if any part of the entry (prompt/completion) contains the keyword (case-insensitive)
                if kw_filter.lower() in entry["prompt"].lower() or \
                   kw_filter.lower() in entry["completion"].lower():
                    matched_by_keyword_filter = True
                    break
            if not matched_by_keyword_filter:
                continue
                
        # Apply cancer type filter (logical OR for multiple selected types)
        if cancer_types and entry["cancer_type"]:
            entry_types = set(ct.strip() for ct in entry["cancer_type"].split(","))
            if not any(ct in entry_types for ct in cancer_types):
                continue
                
        # Apply gene filter (logical OR for multiple selected genes)
        if genes and entry["genes"]:
            entry_genes = set(g.strip() for g in entry["genes"].split(","))
            if not any(g in entry_genes for g in genes):
                continue
                
        filtered.append(result)
    return filtered

# --- LLM API Functions ---

async def call_gemini_api(model_name, chat_history, api_key, generation_config=None):
    """Helper function to call the Gemini API."""
    if not api_key:
        st.error("Gemini API key is not set in Settings.")
        return None

    headers = {'Content-Type': 'application/json'}
    payload = {'contents': chat_history}
    if generation_config:
        payload['generationConfig'] = generation_config

    api_url = f"{GEMINI_API_BASE_URL}{model_name}:generateContent?key={api_key}"

    try:
        # Using requests for synchronous call as Streamlit doesn't natively support async in callbacks
        # For true async in Streamlit, you'd need a separate thread/process or a library like `streamlit-async`
        response = requests.post(api_url, headers=headers, json=payload, timeout=REMOTE_DATA_TIMEOUT)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            st.warning(f"LLM did not return a valid response structure for model {model_name}.")
            return None
    except requests.exceptions.Timeout:
        st.error("LLM API call timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"LLM API call failed: {e}. Check your API key and network connection.")
        return None
    except json.JSONDecodeError:
        st.error("Failed to decode JSON response from LLM API.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM API call: {e}")
        return None

def analyze_query_with_llm(query, available_cancer_types, available_genes, api_key):
    """Uses LLM to suggest cancer types and genes from a query."""
    if not query:
        st.warning("Please enter a query to get AI suggestions.")
        return

    prompt = f"""Analyze the following clinical search query and identify relevant cancer types and genes from the provided lists.
    Prioritize exact matches or very close synonyms.
    
    Query: "{query}"
    
    Available Cancer Types: {', '.join(available_cancer_types)}
    Available Genes: {', '.join(available_genes)}
    
    Return the identified cancer types and genes as a JSON object. If no relevant types or genes are found, return empty arrays.
    
    Example Output:
    ```json
    {{
      "cancer_types": ["Lung Cancer", "NSCLC"],
      "genes": ["EGFR", "ALK"]
    }}
    ```
    """

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    
    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "cancer_types": { "type": "ARRAY", "items": { "type": "STRING" } },
                "genes": { "type": "ARRAY", "items": { "type": "STRING" } }
            }
        }
    }

    with st.spinner("Analyzing query with AI..."):
        json_response_str = call_gemini_api("gemini-2.0-flash", chat_history, api_key, generation_config)
        
        if json_response_str:
            try:
                parsed_json = json.loads(json_response_str)
                st.session_state.llm_suggested_cancer_types = parsed_json.get("cancer_types", [])
                st.session_state.llm_suggested_genes = parsed_json.get("genes", [])
                st.success("AI analysis complete!")
            except json.JSONDecodeError:
                st.error("AI response was not valid JSON. Please try again.")
                st.session_state.llm_suggested_cancer_types = []
                st.session_state.llm_suggested_genes = []
        else:
            st.session_state.llm_suggested_cancer_types = []
            st.session_state.llm_suggested_genes = []

def synthesize_answer_with_llm(query, top_results, api_key):
    """Uses LLM to synthesize an answer from top search results."""
    if not top_results:
        st.session_state.ai_synthesized_answer = "No results to synthesize an answer from."
        return

    context = "\n\n".join([
        f"Question: {res['entry']['prompt']}\nAnswer: {res['entry']['completion']}"
        for res in top_results[:5] # Use top 5 results for context
    ])

    prompt = f"""Based on the following clinical Q&A pairs, synthesize a concise and informative answer to the user's query.
    If the provided context is insufficient to fully answer the query, state that and suggest further search.
    
    User Query: "{query}"
    
    Context Q&A Pairs:
    {context}
    
    Synthesized Answer:
    """

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]

    with st.spinner("Synthesizing answer with AI..."):
        synthesized_text = call_gemini_api("gemini-2.0-flash", chat_history, api_key)
        if synthesized_text:
            st.session_state.ai_synthesized_answer = synthesized_text
            st.success("AI synthesized answer generated!")
        else:
            st.session_state.ai_synthesized_answer = "AI could not synthesize an answer."

# --- Page Layouts ---

def show_settings():
    """Displays the settings page for data source and LLM API key."""
    st.title("⚙️ Settings")
    
    with st.form("network_settings"):
        st.subheader("Data Source Configuration")
        
        use_remote = st.checkbox(
            "Use remote data source",
            value=st.session_state.use_remote_data,
            help="Enable to load data from a remote URL instead of local files."
        )
        
        remote_url = st.text_input(
            "Remote data URL",
            value=st.session_state.remote_data_url,
            placeholder="https://example.com/data.json",
            help="URL to a JSON data file (must be a list of objects with 'prompt' and 'completion')."
        )
        
        st.markdown("---")
        st.subheader("AI Model Configuration (Gemini API)")
        st.warning("Your API key is stored only in your browser's session state and is not saved persistently.")
        
        llm_api_key_input = st.text_input(
            "Gemini API Key",
            value=st.session_state.llm_api_key,
            type="password", # Mask the input
            help="Enter your Google Gemini API key to enable AI features."
        )
        
        if st.form_submit_button("Save Settings"):
            st.session_state.use_remote_data = use_remote
            st.session_state.remote_data_url = remote_url.strip()
            st.session_state.llm_api_key = llm_api_key_input.strip()
            st.success("Settings saved!")
            
            # Clear cache to force reload with new settings
            load_and_index_data.clear()
            st.session_state.data_loaded = False # Force re-load data in main
            
            # Reset LLM suggestions and synthesized answer
            st.session_state.llm_suggested_cancer_types = []
            st.session_state.llm_suggested_genes = []
            st.session_state.ai_synthesized_answer = ""
            
            # Return to home
            st.session_state.show_home = True
            st.session_state.show_settings = False
            st.rerun()
            
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.session_state.show_settings = False
        st.rerun()

def show_home():
    """Displays the home page with search bar and suggestions."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🧬 Precision Cancer Clinical Search")
    with col2:
        if st.button("⚙️ Settings"):
            st.session_state.show_home = False
            st.session_state.show_settings = True
            st.rerun()
            
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials** This tool helps researchers access structured clinical trial information.
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
                    st.session_state.ai_synthesized_answer = "" # Clear previous AI answer
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
            st.session_state.ai_synthesized_answer = "" # Clear previous AI answer
            st.rerun()

def show_results():
    """Displays search results, filters, and LLM interaction options."""
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.session_state.ai_synthesized_answer = "" # Clear previous AI answer
        st.rerun()

    st.title("🔍 Search Results")
    
    # Filters sidebar
    with st.sidebar:
        st.subheader("🔎 Refine Results")
        
        # LLM Query Analysis
        if st.session_state.current_query and st.session_state.llm_api_key:
            if st.button("✨ Get AI Filter Suggestions"):
                analyze_query_with_llm(
                    st.session_state.current_query,
                    st.session_state.cancer_types,
                    st.session_state.genes,
                    st.session_state.llm_api_key
                )
            
            if st.session_state.llm_suggested_cancer_types or st.session_state.llm_suggested_genes:
                st.markdown("**AI Suggested Filters:**")
                if st.session_state.llm_suggested_cancer_types:
                    st.write("Cancer Types:")
                    for ct in st.session_state.llm_suggested_cancer_types:
                        if st.button(f"Add '{ct}'", key=f"ai_add_ct_{ct}"):
                            if ct not in st.session_state.cancer_type_filter:
                                st.session_state.cancer_type_filter.append(ct)
                                st.rerun()
                if st.session_state.llm_suggested_genes:
                    st.write("Genes:")
                    for gene in st.session_state.llm_suggested_genes:
                        if st.button(f"Add '{gene}'", key=f"ai_add_gene_{gene}"):
                            if gene not in st.session_state.gene_filter:
                                st.session_state.gene_filter.append(gene)
                                st.rerun()
                st.markdown("---") # Separator

        # Score filter
        st.session_state.min_score = st.slider(
            "Minimum Match Score",
            min_value=0,
            max_value=200, # Adjusted max value for fuzzy scoring
            value=int(st.session_state.min_score), # Ensure it's an int
            help="Higher scores mean more keywords matched (exact and fuzzy)"
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

    # Load data (from session state if already loaded)
    data = st.session_state.get('data')
    word_index = st.session_state.get('word_index')

    if data is None or word_index is None:
        st.error("Data could not be loaded. Please check settings or local files.")
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
                        st.session_state.ai_synthesized_answer = "" # Clear previous AI answer
                        st.rerun()

    st.markdown(f"**Current Search:** `{st.session_state.current_query}`")

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
    """Displays the filtered search results and options for AI synthesis."""
    st.success(f"Found {len(results)} relevant results (from {len(all_results)} total matches)")
    
    # LLM Answer Synthesis
    if st.session_state.llm_api_key and results:
        if st.button("✨ Synthesize Answer with AI (from top results)"):
            synthesize_answer_with_llm(
                st.session_state.current_query,
                results, # Pass filtered results for context
                st.session_state.llm_api_key
            )
        if st.session_state.ai_synthesized_answer:
            st.subheader("🤖 AI Synthesized Answer:")
            st.info(st.session_state.ai_synthesized_answer)
            st.markdown("---") # Separator

    # Score distribution chart
    if len(results) > 1:
        scores = [r["score"] for r in results]
        st.subheader("Match Score Distribution")
        st.bar_chart(pd.DataFrame({"Score": scores}), use_container_width=True)
        
    # Download buttons for all filtered results
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

    # Display individual results
    st.subheader("Detailed Results:")
    for i, result in enumerate(results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']:.2f} - {entry['prompt'][:70]}...", expanded=(i==1)):
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
            
            # Score details
            with st.expander("🔍 Match Details"):
                st.markdown(f"**Total Score:** {result['score']:.2f}")
                st.markdown(f"**Exact Prompt Matches:** {result.get('exact_prompt_matches', 'N/A')}")
                st.markdown(f"**Exact Completion Matches:** {result.get('exact_completion_matches', 'N/A')}")
                st.markdown(f"**Fuzzy Prompt Score:** {result.get('prompt_fuzzy_score', 'N/A')}")
                st.markdown(f"**Fuzzy Completion Score:** {result.get('completion_fuzzy_score', 'N/A')}")
                if result["matched_keywords"]:
                    st.markdown(f"**Matched Keywords:** {', '.join(result['matched_keywords'])}")
            
            # Download buttons for individual result
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
    """Displays a message when no results are found and provides search tips."""
    st.error("No matches found with current filters. Try these suggestions:")
    
    # Generate suggestions from query
    query_words = set(word.lower() for word in st.session_state.current_query.split() if len(word) > 3)
    suggestions = set()

    if query_words and word_index and data:
        doc_ids = set()
        for word in query_words:
            if word in word_index:
                doc_ids.update(word_index[word])

        for doc_id in list(doc_ids)[:50]: # Limit to 50 potential documents for suggestions
            if doc_id < len(data):
                suggestions.add(data[doc_id]["prompt"])
                if len(suggestions) >= 5: # Show up to 5 suggestions
                    break

    if suggestions:
        st.write("**Similar questions in our database:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion[:50] + "..." if len(suggestion) > 50 else suggestion, 
                             key=f"nores_sugg_{i}"):
                    st.session_state.current_query = suggestion
                    st.session_state.ai_synthesized_answer = "" # Clear previous AI answer
                    st.rerun()
            
    st.markdown("""
    **Search Tips:**
    - Try lowering the minimum score filter.
    - Remove some filters to broaden your search.
    - Check for typos in your search terms (fuzzy search helps with this!).
    - Use more general terms if your search is too specific.
    - Use the 'Get AI Filter Suggestions' button to see if AI can help categorize your query.
    """)

# --- Main App Flow ---

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

    # Load data and suggestions only if not already loaded or if a reload is forced
    if not st.session_state.get('data_loaded', False):
        data, word_index, cancer_types, genes, suggestions = load_and_index_data()
        if data is not None:
            st.session_state.data = data
            st.session_state.word_index = word_index
            st.session_state.suggestions = suggestions
            st.session_state.cancer_types = cancer_types
            st.session_state.genes = genes
            st.session_state.data_loaded = True
        else:
            # If data loading failed, ensure we don't proceed to search/results
            st.session_state.show_home = True # Stay on home or settings if data fails
            st.session_state.show_settings = True # Suggest going to settings to fix data source
            st.warning("Data loading failed. Please check your data files or remote URL in settings.")

    with st.sidebar:
        st.image("https://placehold.co/150x50/ADD8E6/000000?text=Cancer+Search", width=150) # Placeholder image
        st.title("Navigation")

        # Dynamic navigation buttons
        if st.session_state.show_settings:
            if st.button("🏠 Home"):
                st.session_state.show_settings = False
                st.session_state.show_home = True
                st.rerun()
        elif not st.session_state.show_home: # If on results page
            if st.button("🏠 Home"):
                st.session_state.show_home = True
                st.session_state.ai_synthesized_answer = "" # Clear AI answer when going home
                st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This platform helps researchers:
        - Find clinical trial details
        - Access treatment outcomes
        - Download structured data
        - Filter by cancer types and genes
        - Get AI-powered search suggestions and answer synthesis!
        """)

    # Page routing
    if st.session_state.show_settings:
        show_settings()
    elif st.session_state.show_home:
        show_home()
    else:
        # Only show results if data was loaded successfully
        if st.session_state.get('data_loaded', False):
            show_results()
        else:
            st.error("Cannot display results. Data failed to load. Please go to Settings to configure your data source.")
            show_settings() # Redirect to settings if data isn't loaded

if __name__ == "__main__":
    main()
