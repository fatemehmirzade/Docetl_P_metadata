import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

def clean_text_for_json(text: str) -> str:
    """Clean text for JSON compatibility"""
    if not text:
        return ""
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', '  ', text)
    return text.strip()

def extract_section(text: str, section_patterns: list, section_name: str) -> Optional[str]:
    """
    Extract a section from the text based on multiple possible header patterns
    
    Args:
        text: Full text to search
        section_patterns: List of regex patterns for section headers
        section_name: Name of section for debugging
    
    Returns:
        Extracted section text or None if not found
    """
    text_lower = text.lower()
    
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE))
        
        if matches:
            start_pos = matches[0].start()

            end_patterns = [
                r'\n\s*results?\s*\n',
                r'\n\s*\d+\.?\s+results?\s*\n',
                
                r'\n\s*discussion\s*\n',
                r'\n\s*\d+\.?\s+discussion\s*\n',
                r'\n\s*conclusions?\s*\n',
                r'\n\s*\d+\.?\s+conclusions?\s*\n',
                
                r'\n\s*references?\s*\n',
                r'\n\s*bibliography\s*\n',
                r'\n\s*acknowledgements?\s*\n',
                r'\n\s*acknowledgments?\s*\n',
                r'\n\s*funding\s*\n',
                r'\n\s*author\s+contributions?\s*\n',
                r'\n\s*data\s+availability\s*\n',
                r'\n\s*competing\s+interests?\s*\n',
                r'\n\s*ethics\s*\n',
                r'\n\s*availability\s*\n',
                r'\n\s*conflict\s+of\s+interest\s*\n',

                r'\n\s*figures?\s+and\s+tables?\s*\n',
                r'\n\s*supplementary\s+material\s*\n',
                r'\n\s*tables?\s*\n',
                r'\n\s*figures?\s*\n',
                
                r'\n\s*abbreviations?\s*\n',
                r'\n\s*appendix\s*\n',
                r'\n\s*supporting\s+information\s*\n',
            ]

            search_start = start_pos + len(matches[0].group())
            
            end_pos = len(text)
            for end_pattern in end_patterns:
                end_matches = list(re.finditer(end_pattern, text_lower[search_start:], re.IGNORECASE))
                if end_matches:
                    candidate_end = search_start + end_matches[0].start()
                    if candidate_end < end_pos:
                        end_pos = candidate_end
                    break  
            
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) > 50: 
                return section_text
    
    return None

def extract_methods_and_supplementary(text: str) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Extract Methods and Supplementary Information sections from paper text
    
    Returns:
        Tuple of (methods_text, supplementary_text, extraction_info)
    """
    extraction_info = {
        'methods_pattern_used': None,
        'supplementary_pattern_used': None,
        'methods_char_count': 0,
        'supplementary_char_count': 0
    }
    
    methods_patterns = [
        r'\n\s*methods?\s*\n',
        r'\n\s*materials?\s+and\s+methods?\s*\n',
        r'\n\s*experimental\s+procedures?\s*\n',
        r'\n\s*methodology\s*\n',

        r'\n\s*\d+\.?\s+methods?\s*\n',
        r'\n\s*\d+\.?\s+materials?\s+and\s+methods?\s*\n',
        r'\n\s*\d+\.?\s+experimental\s+procedures?\s*\n',

        r'\n\s*experimental\s+design\s*\n',
        r'\n\s*experimental\s+section\s*\n',
        r'\n\s*materials,?\s+methods?,?\s+and\s+protocols?\s*\n',

        r'\n\s*METHODS?\s*\n',
        r'\n\s*MATERIALS?\s+AND\s+METHODS?\s*\n',
        r'\n\s*EXPERIMENTAL\s+PROCEDURES?\s*\n'
    ]
    
    supplementary_patterns = [
        r'\n\s*supplementary\s+information\s*\n',
        r'\n\s*supplementary\s+materials?\s*\n',
        r'\n\s*supplementary\s+methods?\s*\n',
        r'\n\s*supporting\s+information\s*\n',
        r'\n\s*appendix\s*\n',
        r'\n\s*supplementary\s+data\s*\n',

        r'\n\s*supplement\s*\n',
        r'\n\s*supplemental\s+information\s*\n',
        r'\n\s*supplemental\s+materials?\s*\n',
        r'\n\s*supplemental\s+methods?\s*\n',
        r'\n\s*online\s+methods?\s*\n',
        r'\n\s*extended\s+experimental\s+procedures?\s*\n',

        r'\n\s*\d+\.?\s+supplementary\s+information\s*\n',
        
        r'\n\s*SUPPLEMENTARY\s+INFORMATION\s*\n',
        r'\n\s*SUPPLEMENTARY\s+MATERIALS?\s*\n'
    ]
    
    methods = extract_section(text, methods_patterns, "Methods")
    if methods:
        extraction_info['methods_char_count'] = len(methods)
        text_lower = text.lower()
        for i, pattern in enumerate(methods_patterns):
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                extraction_info['methods_pattern_used'] = i
                break
    
    supplementary = extract_section(text, supplementary_patterns, "Supplementary")
    if supplementary:
        extraction_info['supplementary_char_count'] = len(supplementary)
        text_lower = text.lower()
        for i, pattern in enumerate(supplementary_patterns):
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                extraction_info['supplementary_pattern_used'] = i
                break
    
    return methods, supplementary, extraction_info

def prepare_dataset(input_dir: str, output_file: str = "papers_dataset.json"):
    """
    Prepare dataset containing only Methods and Supplementary Information
    
    Args:
        input_dir: Directory containing .txt paper files
        output_file: Output JSON file path
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    data = []
    txt_files = list(input_path.glob("*.txt"))
    
    print(f"Found {len(txt_files)} text files to process\n")
    
    stats = {
        'total': 0,
        'with_methods': 0,
        'with_supplementary': 0,
        'with_both': 0,
        'with_neither': 0
    }
    
    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            text = clean_text_for_json(text)
            
            methods, supplementary, extraction_info = extract_methods_and_supplementary(text)

            stats['total'] += 1
            has_methods = methods is not None
            has_supp = supplementary is not None
            
            if has_methods:
                stats['with_methods'] += 1
            if has_supp:
                stats['with_supplementary'] += 1
            if has_methods and has_supp:
                stats['with_both'] += 1
            if not has_methods and not has_supp:
                stats['with_neither'] += 1
            
            doc = {
                "filename": file_path.name,
                "stem": file_path.stem,
                "methods": clean_text_for_json(methods) if methods else None,
                "supplementary_information": clean_text_for_json(supplementary) if supplementary else None,
                "has_methods": has_methods,
                "has_supplementary": has_supp,
                "extraction_info": extraction_info
            }
            
            data.append(doc)
            
            status = []
            if has_methods:
                status.append(f"Methods ({len(methods)} chars)")
            if has_supp:
                status.append(f"Supplementary ({len(supplementary)} chars)")
            
            status_str = ", ".join(status) if status else "No sections found"
            print(f"{'✓' if (has_methods or has_supp) else '✗'} {file_path.name}: {status_str}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset created: {output_file}")
    print(f"Total documents processed: {stats['total']}")
    print(f"Documents with Methods: {stats['with_methods']} ({stats['with_methods']/stats['total']*100:.1f}%)")
    print(f"Documents with Supplementary: {stats['with_supplementary']} ({stats['with_supplementary']/stats['total']*100:.1f}%)")
    print(f"Documents with both: {stats['with_both']} ({stats['with_both']/stats['total']*100:.1f}%)")
    print(f"Documents with neither: {stats['with_neither']} ({stats['with_neither']/stats['total']*100:.1f}%)")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "text_files_Papers_Ian"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "papers_dataset.json"
    
    prepare_dataset(input_dir, output_file)