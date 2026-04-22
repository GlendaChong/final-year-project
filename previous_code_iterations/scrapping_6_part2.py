import json
import re
from typing import List, Dict, Union, Any

# --- Helper Functions ---

def clean_and_normalize(text: str) -> str:
    """Removes markdown/formatting issues and separates paragraphs with double newlines."""
    text = text.strip()
    # Collapse multiple newlines into two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace soft wraps (single newlines not followed by a newline/space) with a space
    text = re.sub(r'(?<!\n)\n(?![\n\s])', ' ', text)
    return text.strip()

def find_header_start_and_end_index(content: str, header_pattern: str) -> tuple[int, int]:
    """Finds the starting index of the header pattern and the index right after the header line ends."""
    match = re.search(header_pattern, content, re.IGNORECASE | re.DOTALL)
    if not match:
        return -1, -1
    
    start_index = match.start()
    
    # Find the end of the header line (where the text content begins)
    end_of_line_match = re.search(r'[\r\n]', content[start_index:])
    header_end_index = start_index + end_of_line_match.end() if end_of_line_match else start_index + len(match.group(0))
    
    return start_index, header_end_index

def extract_sections_from_content(content: str, paper_id: str) -> Dict[str, str]:
    """
    Extracts Abstract and Introduction using a robust, multi-stage heuristic based on section headers
    and paragraph density.
    """
    
    extracted = {
        'Abstract': 'Not Found',
        'Introduction': 'Not Found',
        'Full_Content': clean_and_normalize(content)
    }

    # --- Define Primary Section Patterns ---
    INTRO_PATTERN = r'[\r\n]1[\.\s]*INTRODUCTION|[\r\n]INTRODUCTION'
    ABSTRACT_PATTERN = r'\s*ABSTRACT\s' 
    # Finds Section 2, 3, 4, 5 or explicit major headers/REFERENCES to set the END boundary.
    NEXT_SECTION_PATTERN = r'[\r\n](2[\.\s][A-Z][a-z\sA-Z]+|3[\.\s][A-Z][a-z\sA-Z]+|4[\.\s][A-Z][a-z\sA-Z]+|5[\.\s][A-Z][a-z\sA-Z]+|REFERENCES)' 
    
    # --- 1. Preparation: Clean and Filter Content Blocks ---
    # Filter out very short lines (metadata/noise, typically less than 5 words) to create ordered narrative units
    all_content_parts = [clean_and_normalize(p) for p in content.strip().split('\n\n') if len(p.split()) > 5]
    raw_content = content
    
    # Find Boundaries based on Headers
    intro_start_idx, intro_content_start_idx = find_header_start_and_end_index(raw_content, INTRO_PATTERN)
    next_section_idx, _ = find_header_start_and_end_index(raw_content, NEXT_SECTION_PATTERN)
    
    # --- 2. Scenario A: Introduction Header Found (Highest Confidence) ---
    if intro_start_idx != -1:
        
        # A. Extract Introduction 
        intro_end_idx = next_section_idx if next_section_idx != -1 else len(raw_content)
        raw_intro = raw_content[intro_content_start_idx:intro_end_idx]
        extracted['Introduction'] = clean_and_normalize(raw_intro)
        
        # B. Extract Abstract (all content before Introduction header)
        raw_abstract_block = raw_content[:intro_start_idx]
        
        abstract_header_match = re.search(ABSTRACT_PATTERN, raw_abstract_block, re.IGNORECASE)
        
        if abstract_header_match:
            raw_abstract_content = raw_abstract_block[abstract_header_match.end():].strip()
        else:
            # Heuristic: Filter preamble blocks (>10 words) and assume Abstract is the last one or two dense blocks.
            preamble_parts = [clean_and_normalize(p) for p in raw_abstract_block.strip().split('\n\n') if len(p.split()) > 10]
            raw_abstract_content = "\n\n".join(preamble_parts[-2:]) 
        
        clean_abstract = clean_and_normalize(raw_abstract_content)
        if clean_abstract:
            extracted['Abstract'] = clean_abstract
            
        return extracted 

    # --- 3. Scenario B: No Introduction Header, But Next Section Found (Positional Boundary) ---
    if next_section_idx != -1:
        # Get the entire preamble block (up to Section 2/3)
        raw_preamble = raw_content[:next_section_idx]
        # Filter paragraphs in the preamble (>10 words)
        preamble_parts = [clean_and_normalize(p) for p in raw_preamble.strip().split('\n\n') if len(p.split()) > 10]
        
        if len(preamble_parts) >= 2:
            # Heuristic: Abstract is the first 1-2 dense blocks, Introduction is the rest.
            abstract_content = "\n\n".join(preamble_parts[:2])
            introduction_content = "\n\n".join(preamble_parts[2:])

            if clean_and_normalize(abstract_content):
                 extracted['Abstract'] = clean_and_normalize(abstract_content)
            if clean_and_normalize(introduction_content):
                 extracted['Introduction'] = clean_and_normalize(introduction_content)
                 
            return extracted 

    # --- 4. Scenario C: NO HEADERS WHATSOEVER (Pure Paragraph Fallback) ---
    if len(all_content_parts) >= 4:
        
        # Heuristic: First block is Abstract, next 3 blocks are Introduction.
        abstract_content = all_content_parts[0]
        introduction_content = "\n\n".join(all_content_parts[1:4]) 

        if clean_and_normalize(abstract_content):
             extracted['Abstract'] = clean_and_normalize(abstract_content)
        if clean_and_normalize(introduction_content):
             extracted['Introduction'] = clean_and_normalize(introduction_content)

    return extracted



def re_extract_sections_from_json(input_filename: str, output_filename: str):
    """
    Loads JSON data, re-runs the advanced extraction logic on failed entries,
    and saves the updated data to a new file.
    """
    try:
        # Load the uploaded file content
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in '{input_filename}'.")
        return

    updated_data = []
    re_extracted_count = 0
    total_re_extracted_attempts = 0

    # Define failure states for Abstract/Introduction from previous runs
    # This captures all possible values from "Not Found" to "Error Processing Content"
    UNSUCCESSFUL_MARKERS = {"Not Found", "Extraction Failed", "Error Processing Content", "Not Extracted"}

    print(f"--- Starting Re-Extraction Process ({len(data)} entries) ---")
    
    for i, entry in enumerate(data):
        
        # Criteria Check: 
        # 1. Has content (Full_Content is available and not a marker) AND
        # 2. Abstract OR Introduction failed previously (is in the UNSUCCESSFUL_MARKERS set)
        has_full_content = entry.get('Full_Content') not in UNSUCCESSFUL_MARKERS and len(entry.get('Full_Content', '')) > 200
        is_unextracted = entry.get('Abstract') in UNSUCCESSFUL_MARKERS or entry.get('Introduction') in UNSUCCESSFUL_MARKERS

        if has_full_content and is_unextracted:
            
            total_re_extracted_attempts += 1
            content = entry['Full_Content']
            # Use 'id' from the entry or its index if 'id' is missing
            paper_id = entry.get('id', i)
            
            # Apply the advanced heuristic extraction
            re_extracted = extract_sections_from_content(content, paper_id)
            
            # Update the entry only if the new result is an improvement (not a failure marker)
            current_improvement = False
            
            if re_extracted['Abstract'] not in UNSUCCESSFUL_MARKERS:
                entry['Abstract'] = re_extracted['Abstract']
                current_improvement = True
            
            if re_extracted['Introduction'] not in UNSUCCESSFUL_MARKERS:
                entry['Introduction'] = re_extracted['Introduction']
                current_improvement = True
            
            # Log the result
            if current_improvement:
                 print(f"✅ Re-extracted ID {paper_id}: Abstract improved? {re_extracted['Abstract'] not in UNSUCCESSFUL_MARKERS}. Intro improved? {re_extracted['Introduction'] not in UNSUCCESSFUL_MARKERS}.")
                 re_extracted_count += 1
            else:
                 print(f"❌ Re-extraction failed to improve for ID {paper_id}. (Still Not Found)")

        updated_data.append(entry)

    # Save the updated data
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=4)
        print(f"\n--- Process Complete ---")
        print(f"Attempted re-extraction on {total_re_extracted_attempts} failed entries.")
        print(f"Total entries showing improvement: {re_extracted_count}")
        print(f"Updated data saved to '{output_filename}'")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Define the file names
    INPUT_FILE = 'extracted_papers_summary_5.json'
    OUTPUT_FILE = 're_extracted_papers_summary_5.json'

    # Execute the main function
    re_extract_sections_from_json(INPUT_FILE, OUTPUT_FILE)