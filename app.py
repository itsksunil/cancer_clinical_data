# Update the load_and_index_data function to handle nct_id field
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
        nct_numbers = set()

        for idx, entry in enumerate(raw_data):
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                # Clean and standardize data
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                
                # Extract NCT numbers from nct_id field if present
                found_ncts = []
                if "nct_id" in entry:
                    nct_id = str(entry["nct_id"]).strip()
                    if nct_id.startswith("NCT") and len(nct_id) == 11 and nct_id[3:].isdigit():
                        found_ncts.append(nct_id)
                        nct_numbers.add(nct_id)
                
                # Also search for NCT numbers in text as fallback
                text = f"{entry['prompt']} {entry['completion']}"
                for word in text.split():
                    if word.startswith("NCT") and len(word) == 11 and word[3:].isdigit():
                        if word not in found_ncts:
                            found_ncts.append(word)
                            nct_numbers.add(word)
                
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
                    "nct_numbers": found_ncts,
                    "nct_id": entry.get("nct_id", ""),
                    "id": idx
                })
                all_prompts.append(entry["prompt"])

                # Index words with special handling for NCT numbers
                text = f"{entry['prompt']} {entry['completion']}".lower()
                words = set()
                for word in text.split():
                    # Keep NCT numbers as-is (case-sensitive)
                    if word.startswith("NCT") and len(word) == 11 and word[3:].isdigit():
                        words.add(word)  # Keep original case
                        word_index[word].append(idx)
                        all_keywords.add(word)
                    else:
                        # Normal word processing
                        word = word.strip(".,!?;:-_'\"()[]{}")
                        if len(word) > 2 and word.isalpha():
                            words.add(word.lower())
                            all_keywords.add(word.lower())
                
                # Also index the nct_id separately if it exists
                if "nct_id" in entry and entry["nct_id"]:
                    nct = entry["nct_id"]
                    word_index[nct].append(idx)
                    all_keywords.add(nct)
                
                for word in words:
                    word_index[word].append(idx)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None, [], [], [], [], []
        
        # Generate random suggestions
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions, sorted(all_keywords), sorted(nct_numbers)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, [], [], [], [], []

# Enhance the keyword_search function for better NCT number handling
def keyword_search(query: str, dataset: List[Dict], word_index: Dict[str, List[int]]) -> List[Dict]:
    if not query or not dataset:
        return []
    
    # Check if query is an NCT number (case-sensitive)
    is_nct_search = query.startswith("NCT") and len(query) == 11 and query[3:].isdigit()
    
    if is_nct_search:
        # First check exact NCT ID matches
        exact_nct_matches = []
        for doc_id in word_index.get(query, []):
            if doc_id < len(dataset):
                entry = dataset[doc_id]
                # Give higher score if it matches the nct_id field specifically
                score = 200 if entry.get("nct_id", "") == query else 100
                exact_nct_matches.append({
                    "entry": entry,
                    "score": score,
                    "exact_matches": 1,
                    "phrase_bonus": 0,
                    "prompt_matches": 1 if query in entry["prompt"] else 0,
                    "matched_keywords": {query},
                    "doc_id": doc_id,
                    "is_nct_match": True
                })
        
        # If we found exact NCT ID matches, return them immediately
        if exact_nct_matches:
            return exact_nct_matches
        
        # Fallback to search in text if no exact NCT ID matches
        doc_ids = set()
        for word in word_index:
            if word.startswith("NCT") and query in word:  # Partial NCT match
                doc_ids.update(word_index[word])
        
        partial_matches = []
        for doc_id in doc_ids:
            if doc_id < len(dataset):
                entry = dataset[doc_id]
                # Calculate how closely the NCT numbers match
                nct_similarity = max(
                    (len(query) - levenshtein_distance(query, nct)) / len(query)
                    for nct in entry["nct_numbers"]
                )
                score = int(50 * nct_similarity)  # Score based on similarity
                partial_matches.append({
                    "entry": entry,
                    "score": score,
                    "exact_matches": 1 if query in entry["nct_numbers"] else 0,
                    "phrase_bonus": 0,
                    "prompt_matches": 1 if query in entry["prompt"] else 0,
                    "matched_keywords": {n for n in entry["nct_numbers"] if query in n},
                    "doc_id": doc_id,
                    "is_nct_match": True
                })
        
        return exact_nct_matches + partial_matches
    
    # Normal keyword search processing...
    # [Rest of your existing keyword_search function]
    
    return ranked_results

# Add Levenshtein distance for partial NCT matching
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# Update the display_results function to highlight NCT matches
def display_results(results: List[Dict], all_results: List[Dict]):
    st.success(f"Found {len(results)} relevant results (from {len(all_results)} total matches)")
    
    # Group results by NCT number if this was an NCT search
    if results and results[0].get("is_nct_match", False):
        nct_groups = defaultdict(list)
        for result in results:
            for nct in result["entry"]["nct_numbers"]:
                nct_groups[nct].append(result)
        
        # Display each NCT group separately
        for nct, group in nct_groups.items():
            st.subheader(f"Clinical Trial: {nct}")
            for i, result in enumerate(group, 1):
                display_single_result(result, i)
    else:
        # Normal display
        for i, result in enumerate(results, 1):
            display_single_result(result, i)

def display_single_result(result: Dict, result_num: int):
    entry = result["entry"]
    with st.expander(f"#{result_num} | Score: {result['score']} - {entry['prompt'][:50]}...", expanded=(result_num==1)):
        # Highlight NCT numbers in prompt
        prompt_display = entry["prompt"]
        for nct in entry["nct_numbers"]:
            prompt_display = prompt_display.replace(nct, f"**{nct}**")
        st.markdown(f"**Question:** {prompt_display}")
        
        # Highlight in completion
        completion_display = entry["completion"]
        for nct in entry["nct_numbers"]:
            completion_display = completion_display.replace(nct, f"**{nct}**")
        st.markdown(f"**Answer:** {completion_display}")
        
        # Show NCT numbers prominently if present
        if entry["nct_numbers"]:
            st.markdown(f"**Clinical Trial IDs:** {', '.join(entry['nct_numbers'])}")
        
        # Rest of your display code...
