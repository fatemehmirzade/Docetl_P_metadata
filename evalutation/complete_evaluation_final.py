import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
from fuzzywuzzy import fuzz
import json

try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval
    DEEPEVAL_AVAILABLE = True
    print(" DeepEval available for LLM-as-judge evaluation")
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print(" DeepEval not installed. Install with: pip install deepeval")
    print("  LLM-as-judge evaluation will be skipped")

GROUND_TRUTH_DIR = "/Users/fateme/Downloads/Ians_Annotations"
PREDICTIONS_DIR = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/annotations"
RESULTS_DIR = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/evaluation_results_test"
PAPERS_JSON = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/papers_dataset.json"

HALLUCINATION_THRESHOLD = 85

USE_DEEPEVAL_LLM_JUDGE = True
EVALUATION_MODEL = "gpt-4o"  # Model to use for LLM-as-judge

EXCLUDED_TYPES = {
    'time', 
    'temperature', 
    'compound', 
    'concentrationofcompound',
    'factorvalue'  # Also catches FactorValue[X] variants
}

os.makedirs(RESULTS_DIR, exist_ok=True)

class ProteomicsCompletenessMetric(GEval):
    """
    Custom metric to evaluate completeness of proteomics metadata extraction.
    Checks if all entity types describing the proteomics/MS experiment are covered.
    """
    def __init__(self, threshold: float = 0.7, model: str = None):
        if model is None:
            model = EVALUATION_MODEL

        super().__init__(
            name="Proteomics_Completeness",
            model=model,
            criteria="""
            You are evaluating the COMPLETENESS of proteomics metadata extraction.

            TASK:
            1. Read the INPUT (JSON with abstract, methods, supplementary sections)
            2. Find the parts that describe HOW the proteomics or mass spectrometry experiment was conducted
            3. Look at the ACTUAL OUTPUT (extracted annotations by entity type)
            4. Check: Does the extraction COVER all the entity types that should be present?

            WHAT TO CHECK:
            - For all entity types that are described in the experiment, are they extracted?
            - Are there missing entity types? (e.g., the text mentions an organism but no Organism was extracted)
            - Are there missing entity types? (e.g., the text describes an instrument but no Instrument was extracted)
            - Does the extraction cover the full scope of the experiment description?

            IMPORTANT:
            - Focus on COVERAGE: Did they extract all the types of information mentioned?
            - Don't worry about exact values (that's for accuracy)
            - Look for MISSING entity types that should have been extracted

            EXAMPLES:
            - If methods say "HeLa cells were analyzed" but no CellLine is extracted → INCOMPLETE
            - If methods describe "Q Exactive mass spectrometer" but no Instrument is extracted → INCOMPLETE  
            - If sample prep details are mentioned but no CleavageAgent/Label extracted → INCOMPLETE
            - If all mentioned entity types are extracted (even if not all values) → COMPLETE

            SCORING:
            Score 0.9-1.0: All entity types mentioned in experiment are extracted
            Score 0.7-0.8: Most entity types extracted, only minor types missing
            Score 0.5-0.6: Several important entity types missing
            Score 0.0-0.4: Many entity types not extracted, major gaps
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold=threshold,
            evaluation_steps=[
                "Read the input JSON and find proteomics/MS experiment description",
                "List all entity types that SHOULD be present based on what's described",
                "Check the actual output to see which entity types ARE present",
                "Identify any MISSING entity types",
                "Score based on coverage of entity types"
            ]
        )

class ProteomicsAccuracyMetric(GEval):
    """
    Custom metric to evaluate accuracy of proteomics metadata extraction.
    Checks if extracted values exist in source text and are extracted correctly.
    """
    def __init__(self, threshold: float = 0.8, model: str = None):
        if model is None:
            model = EVALUATION_MODEL

        super().__init__(
            name="Proteomics_Accuracy",
            model=model,
            criteria="""
            You are evaluating the ACCURACY of proteomics metadata extraction.

            TASK:
            1. Read the INPUT (JSON with abstract, methods, supplementary sections)
            2. Find the parts describing the proteomics or mass spectrometry experiment
            3. Look at each extracted annotation in ACTUAL OUTPUT
            4. For EACH extracted value, check two things:
               a) Does this value actually EXIST in the input text?
               b) Is it extracted exactly as mentioned (same wording/terminology)?

            WHAT TO CHECK:

            1. EXISTENCE: Does the extracted value appear in the input?
               - Search the input text for each extracted value
               - It should be present (exact match or very close paraphrase)
               - If you cannot find it in the input → HALLUCINATION (major penalty)

            2. EXTRACTION FIDELITY: Is it extracted the same way it was mentioned?
               - If input says "Q Exactive HF", extraction should say "Q Exactive HF" (not "Q Exactive")
               - If input says "trypsin", extraction should say "trypsin" (not "enzymatic digestion")
               - If input says "HeLa cells", extraction should say "HeLa cells" (not just "HeLa" or "cells")
               - Numbers, units, and specific terms should be preserved exactly

            IMPORTANT:
            - Every extracted value must be verifiable in the input text
            - Hallucinations (values not in input) are SERIOUS accuracy problems
            - Simplifications or paraphrases reduce accuracy
            - Focus on whether extraction matches the SOURCE, not whether it's scientifically correct

            EXAMPLES:
            - Input: "digested with trypsin" → Extract: "trypsin"  ACCURATE
            - Input: "digested with trypsin" → Extract: "protease"  INACCURATE (not same wording)
            - Input: "Orbitrap Fusion Lumos" → Extract: "Orbitrap Fusion Lumos"  ACCURATE
            - Input: "Orbitrap Fusion Lumos" → Extract: "Orbitrap"  INACCURATE (incomplete)
            - Input: mentions "mouse" → Extract: "Mus musculus"  ACCURATE (valid scientific name)
            - Input: no mention of organism → Extract: "Homo sapiens"  HALLUCINATION

            SCORING:
            Score 0.9-1.0: All values exist in source, extracted exactly as written
            Score 0.7-0.8: All values exist, minor wording differences but meaning preserved
            Score 0.5-0.6: Some values missing from source or significantly paraphrased
            Score 0.0-0.4: Many hallucinations or incorrectly extracted values
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold=threshold,
            evaluation_steps=[
                "Read input JSON to find proteomics/MS experiment description",
                "For each value in actual output, search for it in the input",
                "Check: Does this value exist in the input text?",
                "Check: Is it extracted with the same wording/terminology?",
                "Count hallucinations and extraction errors",
                "Calculate accuracy score"
            ]
        )

class ProteomicsConsistencyMetric(GEval):
    """
    Custom metric to evaluate consistency of proteomics metadata extraction.
    Checks if extracted values represent a coherent, realistic proteomics/MS experiment.
    """
    def __init__(self, threshold: float = 0.7, model: str = None):
        if model is None:
            model = EVALUATION_MODEL

        super().__init__(
            name="Proteomics_Consistency",
            model=model,
            criteria="""
            You are evaluating the CONSISTENCY of proteomics metadata extraction.

            TASK:
            1. Read the INPUT (JSON describing a proteomics/MS experiment)
            2. Look at the ACTUAL OUTPUT (extracted annotations)
            3. Ask: Does this extraction represent a REAL, COHERENT proteomics or mass spectrometry experiment?

            WHAT TO CHECK:

            1. EXPERIMENTAL COHERENCE: Do the pieces fit together?
               - If Organism is "mouse", does CellLine make sense? (e.g., "MEF cells" fits, "HeLa" doesn't)
               - If Instrument is a specific MS model, do the associated details match? (e.g., Orbitrap → HCD fragmentation makes sense)
               - Does the sample prep workflow make sense? (Label → Digestion → Separation → MS)

            2. SCIENTIFIC VALIDITY: Are the extracted values realistic for proteomics?
               - Are the instruments real mass spectrometers? (e.g., "Q Exactive" , "HPLC machine" )
               - Are the reagents appropriate? (e.g., "trypsin" for CleavageAgent , "water" )
               - Are cell lines real? (e.g., "HeLa" , "random cells" )
               - Do the methods align with standard proteomics workflows?

            3. INTERNAL CONSISTENCY: Are there contradictions?
               - Multiple incompatible organisms listed?
               - Instruments that don't match the described method (e.g., MALDI for LC-MS/MS)?
               - Conflicting information (e.g., "label-free" Label but also "TMT" Label)?

            4. COMPLETENESS OF WORKFLOW: Does it describe a viable experiment?
               - Core elements present: Sample source → Preparation → Analysis → Detection
               - Not just random scattered facts
               - Enough information to understand what was done

            IMPORTANT:
            - This is about whether the extraction makes SCIENTIFIC SENSE
            - Not comparing to ground truth - evaluating internal logic
            - Focus on whether a proteomics researcher would find this coherent

            EXAMPLES:
            - Mouse + MEF cells + Q Exactive + trypsin + LC-MS/MS → CONSISTENT 
            - Human + mouse + rat all as Organism → INCONSISTENT (unless explicitly comparative study)
            - Orbitrap Fusion + MALDI ionization → INCONSISTENT (Orbitrap uses ESI)
            - Label: "TMT" and Label: "label-free" together → INCONSISTENT
            - Only Instrument extracted, no sample info → INCOMPLETE/INCONSISTENT

            SCORING:
            Score 0.9-1.0: Fully coherent, realistic proteomics experiment, all pieces fit
            Score 0.7-0.8: Mostly coherent, minor inconsistencies or gaps
            Score 0.5-0.6: Some contradictions or unrealistic elements
            Score 0.0-0.4: Major inconsistencies, doesn't represent a viable experiment
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold=threshold,
            evaluation_steps=[
                "Read input to understand the experiment described",
                "Review all extracted annotations in actual output",
                "Check if annotations form a coherent experimental workflow",
                "Verify scientific validity of extracted values",
                "Look for internal contradictions or incompatibilities",
                "Assess whether this represents a realistic proteomics experiment",
                "Calculate consistency score"
            ]
        )

def evaluate_with_deepeval_judge(paper_id, predicted_anns, ground_truth_anns, source_text):
    """
    Evaluate a single paper using DeepEval LLM-as-judge metrics.

    Args:
        paper_id: Paper identifier
        predicted_anns: Dictionary of predicted annotations {entity_type: [values]}
        ground_truth_anns: Dictionary of ground truth annotations (not used, for compatibility)
        source_text: Full source text from the paper (JSON with sections)

    Returns:
        Dictionary with scores and explanations
    """
    if not DEEPEVAL_AVAILABLE or not USE_DEEPEVAL_LLM_JUDGE:
        return {}

    if not source_text or not predicted_anns:
        return {}

    try:
        predicted_str = json.dumps(predicted_anns, indent=2, ensure_ascii=False)

        test_case = LLMTestCase(
            input=source_text,
            actual_output=predicted_str
        )

        completeness_metric = ProteomicsCompletenessMetric(
            threshold=0.7,
            model=EVALUATION_MODEL
        )

        accuracy_metric = ProteomicsAccuracyMetric(
            threshold=0.8,
            model=EVALUATION_MODEL
        )

        consistency_metric = ProteomicsConsistencyMetric(
            threshold=0.7,
            model=EVALUATION_MODEL
        )

        completeness_metric.measure(test_case)
        accuracy_metric.measure(test_case)
        consistency_metric.measure(test_case)

        return {
            'completeness_score': completeness_metric.score,
            'completeness_reason': completeness_metric.reason,
            'accuracy_score': accuracy_metric.score,
            'accuracy_reason': accuracy_metric.reason,
            'consistency_score': consistency_metric.score,
            'consistency_reason': consistency_metric.reason
        }

    except Exception as e:
        print(f"   DeepEval evaluation failed for {paper_id}: {e}")
        return {}

def parse_ann_file(file_path):
    """
    Parse a .ann file in the format: EntityType: value
    Returns list of tuples: (entity_type, text, line_num)
    """
    entities = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        entity_type = parts[0].strip()
                        text = parts[1].strip()
                        entities.append((entity_type, text, line_num))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return entities

def should_exclude_entity(entity_type):
    """
    Check if entity type should be excluded.
    Only excludes if the base entity type (after removing FactorValue prefix) 
    matches one of the excluded types.
    """
    normalized = normalize_entity_type(entity_type)
    normalized_lower = normalized.lower().strip()

    if normalized_lower in EXCLUDED_TYPES:
        return True

    entity_lower = entity_type.lower().strip()
    if entity_lower.startswith('factorvalue'):
        return True

    return False

def normalize_entity_type(entity_type):
    """Normalize entity type for comparison."""
    if '[' in entity_type:
        entity_type = entity_type.split('[')[1].rstrip(']')
    return entity_type

def normalize_text(text):
    """Normalize text for better comparison."""
    text = ' '.join(text.split())
    text = text.lower().strip()
    return text

def load_papers_dataset(json_path):
    """Load papers dataset JSON to get source text for hallucination detection."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        papers_dict = {}
        for paper in papers:
            paper_id = paper.get('filename', paper.get('stem', ''))
            if paper_id:
                paper_key = paper_id.replace('.txt', '')
                papers_dict[paper_key] = {
                    'abstract': paper.get('abstract', ''),
                    'methods': paper.get('methods', ''),
                    'supplementary': paper.get('supplementary', '')
                }

        return papers_dict
    except Exception as e:
        print(f"Warning: Could not load papers dataset: {e}")
        return {}

def check_hallucination(predicted_value, source_text, threshold=85):
    """
    Check if a predicted value appears in the source text.
    Returns: (is_found, confidence_score)
    """
    if not source_text or not predicted_value:
        return False, 0.0

    pred_norm = ' '.join(predicted_value.split()).lower().strip()
    source_norm = ' '.join(source_text.split()).lower()

    if pred_norm in source_norm:
        return True, 1.0

    pred_clean = pred_norm.replace('β', 'b').replace('α', 'a').replace('γ', 'g')
    pred_clean = pred_clean.replace('µ', 'u').replace('°', '')

    source_clean = source_norm.replace('β', 'b').replace('α', 'a').replace('γ', 'g')
    source_clean = source_clean.replace('µ', 'u').replace('°', '')

    if pred_clean in source_clean:
        return True, 0.95

    words = source_norm.split()
    max_score = 0

    pred_words = pred_norm.split()
    for window_size in [len(pred_words), len(pred_words) + 2, len(pred_words) + 5, len(pred_words) + 10]:
        if window_size > len(words):
            continue
        for i in range(len(words) - window_size + 1):
            chunk = ' '.join(words[i:i + window_size])
            score = fuzz.partial_ratio(pred_norm, chunk)
            max_score = max(max_score, score)

            chunk_clean = ' '.join(chunk.split())
            score_clean = fuzz.partial_ratio(pred_clean, chunk_clean.replace('β', 'b').replace('α', 'a').replace('µ', 'u'))
            max_score = max(max_score, score_clean)

            if max_score >= threshold:
                return True, max_score / 100.0

    return max_score >= threshold, max_score / 100.0

ABBREVIATIONS = {
    'dtt': 'dithiothreitol',
    'iaa': 'iodoacetamide',
    'tcep': 'tris(2-carboxyethyl)phosphine',
    'tris(2-carboxyethyl)phosphine': 'tcep',
    'tfa': 'trifluoroacetic acid',
    'acn': 'acetonitrile',
    'meoh': 'methanol',
    'etoh': 'ethanol',
    'fa': 'formic acid',

    'cells': 'cell',
    'cell line': 'cell',
    'cell lines': 'cell',
    'hesc': 'human embryonic stem cell',
    'hescs': 'human embryonic stem cell',
    'esc': 'embryonic stem cell',
    'escs': 'embryonic stem cell',
    'msc': 'mesenchymal stem cell',
    'mscs': 'mesenchymal stem cell',
    'bm-msc': 'bone marrow mesenchymal stem cell',
    'bone marrow-derived msc': 'bone marrow mesenchymal stem cell',
    'ips': 'induced pluripotent stem cell',
    'ipsc': 'induced pluripotent stem cell',
    'ipscs': 'induced pluripotent stem cell',

    'human': 'homo sapiens',
    'humans': 'homo sapiens',
    'mouse': 'mus musculus',
    'mice': 'mus musculus',
    'rat': 'rattus norvegicus',
    'rats': 'rattus norvegicus',

    'male': 'male',
    'men': 'male',
    'man': 'male',
    'female': 'female',
    'women': 'female',
    'woman': 'female',

    'dda': 'data dependent acquisition',
    'data-dependent acquisition': 'data dependent acquisition',
    'dia': 'data independent acquisition',
    'data-independent acquisition': 'data independent acquisition',
    'srm': 'selected reaction monitoring',
    'mrm': 'multiple reaction monitoring',
    'prm': 'parallel reaction monitoring',
    'ap-ms': 'affinity purification mass spectrometry',
    'tap-ms': 'tandem affinity purification mass spectrometry',
    'tap': 'tandem affinity purification',

    'hplc': 'high performance liquid chromatography',
    'lc': 'liquid chromatography',
    'uplc': 'ultra performance liquid chromatography',
    'rp-hplc': 'reverse phase high performance liquid chromatography',

    'ms': 'mass spectrometry',
    'ms/ms': 'tandem mass spectrometry',
    'lc-ms': 'liquid chromatography mass spectrometry',
    'lc-ms/ms': 'liquid chromatography tandem mass spectrometry',

    'hcd': 'higher energy collisional dissociation',
    'higher-energy collisional dissociation': 'higher energy collisional dissociation',
    'cid': 'collision induced dissociation',
    'collision-induced dissociation': 'collision induced dissociation',
    'etd': 'electron transfer dissociation',
    'electron-transfer dissociation': 'electron transfer dissociation',
    'ecd': 'electron capture dissociation',
    'electron-capture dissociation': 'electron capture dissociation',

    'esi': 'electrospray ionization',
    'maldi': 'matrix assisted laser desorption ionization',
    'apci': 'atmospheric pressure chemical ionization',
    'appi': 'atmospheric pressure photoionization',

    'qtof': 'quadrupole time of flight',
    'q-tof': 'quadrupole time of flight',
    'tof': 'time of flight',
    'ft-icr': 'fourier transform ion cyclotron resonance',
    'fticr': 'fourier transform ion cyclotron resonance',
    'orbitrap': 'orbitrap',
    'it': 'ion trap',
    'lit': 'linear ion trap',

    'silac': 'stable isotope labeling amino acids cell culture',
    'stable isotope labeling by amino acids in cell culture': 'stable isotope labeling amino acids cell culture',
    'tmt': 'tandem mass tag',
    'tandem mass tags': 'tandem mass tag',
    'itraq': 'isobaric tags relative absolute quantitation',
    'isobaric tags for relative and absolute quantitation': 'isobaric tags relative absolute quantitation',
    'lfq': 'label free',
    'label free': 'lfq',
    'label-free': 'lfq',
    'label free quantification': 'lfq',
    'label-free quantification': 'lfq',

    'carbamidomethylation': 'carbamidomethyl',
    'carbamidomethyl': 'carbamidomethylation',
    'oxidation': 'oxidized',
    'phosphorylation': 'phosphorylated',
    'acetylation': 'acetylated',
    'methylation': 'methylated',
    'ubiquitination': 'ubiquitinated',

    'trypsin': 'trypsin',
    'lysc': 'lys-c',
    'lys-c': 'lysc',
    'argc': 'arg-c',
    'arg-c': 'argc',
    'gluc': 'glu-c',
    'glu-c': 'gluc',
    'chymotrypsin': 'chymotrypsin',

    'samples': 'sample',
    'replicates': 'replicate',
    'biological replicates': 'biological replicate',
    'technical replicates': 'technical replicate',
    'proteins': 'protein',
    'peptides': 'peptide',
    'molecules': 'molecule',
    'compounds': 'compound',
    'organisms': 'organism',
    'tissues': 'tissue',
    'fractions': 'fraction',
    'modifications': 'modification',
    'treatments': 'treatment',

    'tissue': 'tissue',
    'tissues': 'tissue',
    'plasma': 'plasma',
    'serum': 'serum',
    'lysate': 'lysate',
    'lysates': 'lysate',
    'cell lysate': 'lysate',
    'whole cell lysate': 'lysate',
}

def expand_abbreviations(text):
    """Expand common abbreviations in text."""
    text = normalize_text(text)

    if text in ABBREVIATIONS:
        return ABBREVIATIONS[text]

    expanded = text
    for abbr, full in ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expanded = re.sub(pattern, full, expanded)

    return expanded

def normalize_for_semantic_matching(text):
    """Apply semantic normalization for common scientific terms."""
    text = normalize_text(text)

    text = expand_abbreviations(text)

    stop_words = ['a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with']
    words = text.split()
    words = [w for w in words if w not in stop_words or len(words) <= 3]
    text = ' '.join(words)

    return text

def semantic_similarity(text1, text2):
    """Check if texts are semantically equivalent."""
    norm1 = normalize_for_semantic_matching(text1)
    norm2 = normalize_for_semantic_matching(text2)

    if norm1 == norm2:
        return True

    if len(norm1) > 2 and len(norm2) > 2:
        if norm1 in norm2 or norm2 in norm1:
            return True

    return False

def extract_tokens(text):
    """Extract meaningful tokens from text."""
    text = normalize_text(text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if len(t) > 1 or t in ['c', 'q']]  # Keep some single letters
    return set(tokens)

def extract_key_terms(text):
    """Extract key scientific terms and abbreviations."""
    text = normalize_text(text)

    abbrevs = re.findall(r'\(([A-Z0-9\-]+)\)', text)

    key_terms = set()

    for abbr in abbrevs:
        key_terms.add(abbr.lower())

    tokens = extract_tokens(text)
    key_terms.update(tokens)

    return key_terms

def substring_match(text1, text2):
    """Check if one text is a substantial substring of the other."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if len(norm1) < len(norm2):
        shorter, longer = norm1, norm2
    else:
        shorter, longer = norm2, norm1

    if len(shorter) < 3:
        return False

    return shorter in longer

def token_overlap_ratio(text1, text2):
    """Calculate token overlap ratio (Jaccard similarity)."""
    tokens1 = extract_tokens(text1)
    tokens2 = extract_tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0

def key_terms_match(text1, text2):
    """Check if key terms match between texts."""
    terms1 = extract_key_terms(text1)
    terms2 = extract_key_terms(text2)

    if not terms1 or not terms2:
        return 0.0

    intersection = len(terms1 & terms2)
    smaller_set = min(len(terms1), len(terms2))

    return intersection / smaller_set if smaller_set > 0 else 0.0

def sequence_similarity(text1, text2):
    """Calculate sequence similarity."""
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()

def calculate_match_score(text1, text2):
    """
    Calculate comprehensive match score.
    Returns score between 0 and 1.
    """
    if normalize_text(text1) == normalize_text(text2):
        return 1.0

    if semantic_similarity(text1, text2):
        return 0.95

    if substring_match(text1, text2):
        return 0.9

    token_score = token_overlap_ratio(text1, text2)
    key_terms_score = key_terms_match(text1, text2)
    seq_score = sequence_similarity(text1, text2)

    combined_score = (0.4 * key_terms_score + 
                     0.35 * token_score + 
                     0.25 * seq_score)

    return combined_score

def calculate_metrics_per_type(gt_entities_by_type, pred_entities_by_type):
    """
    Calculate precision, recall, and F1 for each entity type.
    Uses many-to-many fuzzy matching:
    - Each GT can match multiple predictions
    - Each prediction can match multiple GTs
    - CRITICAL: Only matches within the same file

    Entity format: (type, text, line_num, filename)
    """
    results = {}

    for gt_type in sorted(gt_entities_by_type.keys()):
        gt_entities = gt_entities_by_type[gt_type]
        pred_entities_same_type = pred_entities_by_type.get(gt_type, [])

        if not pred_entities_same_type:
            results[gt_type] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(gt_entities),
                'gt_count': len(gt_entities),
                'pred_count': 0,
                'matches': [],
                'unmatched_gt': [{'gt_text': txt, 'gt_line': ln, 'gt_file': fn} 
                                for _, txt, ln, fn in gt_entities],
                'unmatched_pred': []
            }
            continue

        gt_matches = defaultdict(list)
        pred_matches = defaultdict(list)

        threshold = 0.5

        for gt_idx, (_, gt_text, gt_line, gt_file) in enumerate(gt_entities):
            for pred_idx, (_, pred_text, pred_line, pred_file) in enumerate(pred_entities_same_type):
                if gt_file != pred_file:
                    continue

                score = calculate_match_score(gt_text, pred_text)

                if score >= threshold:
                    gt_matches[gt_idx].append((pred_idx, score, pred_text, pred_line, pred_file))
                    pred_matches[pred_idx].append((gt_idx, score, gt_text, gt_line, gt_file))

        tp = len(gt_matches)

        fn = len(gt_entities) - len(gt_matches)

        fp = len(pred_entities_same_type) - len(pred_matches)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        matches = []
        for gt_idx, match_list in gt_matches.items():
            _, gt_text, gt_line, gt_file = gt_entities[gt_idx]
            for pred_idx, score, pred_text, pred_line, pred_file in match_list:
                matches.append({
                    'gt_text': gt_text,
                    'gt_line': gt_line,
                    'gt_file': gt_file,
                    'pred_text': pred_text,
                    'pred_line': pred_line,
                    'pred_file': pred_file,
                    'similarity': score,
                    'match_type': get_match_type(gt_text, pred_text, score)
                })

        unmatched_gt = []
        for gt_idx, (_, gt_text, gt_line, gt_file) in enumerate(gt_entities):
            if gt_idx not in gt_matches:
                unmatched_gt.append({'gt_text': gt_text, 'gt_line': gt_line, 'gt_file': gt_file})

        unmatched_pred = []
        for pred_idx, (_, pred_text, pred_line, pred_file) in enumerate(pred_entities_same_type):
            if pred_idx not in pred_matches:
                unmatched_pred.append({'pred_text': pred_text, 'pred_line': pred_line, 'pred_file': pred_file})

        results[gt_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_count': len(gt_entities),
            'pred_count': len(pred_entities_same_type),
            'matches': matches,
            'unmatched_gt': unmatched_gt,
            'unmatched_pred': unmatched_pred
        }

    return results

def get_match_type(text1, text2, score):
    """Determine the type of match for reporting."""
    if normalize_text(text1) == normalize_text(text2):
        return "Exact"
    elif semantic_similarity(text1, text2):
        return "Semantic"
    elif substring_match(text1, text2):
        return "Substring"
    elif score >= 0.9:
        return "Near-Exact"
    elif score >= 0.7:
        return "High-Overlap"
    else:
        return "Partial"

def evaluate_files(gt_dir, pred_dir):
    """Evaluate all matching .ann files."""
    gt_files = {f.stem: f for f in Path(gt_dir).glob('*.ann')}
    pred_files = {f.stem: f for f in Path(pred_dir).glob('*.ann')}

    common_files = set(gt_files.keys()) & set(pred_files.keys())

    if not common_files:
        print("No matching .ann files found!")
        return None

    print(f"Found {len(common_files)} matching files")

    print("Loading papers dataset for hallucination detection...")
    papers_dict = load_papers_dataset(PAPERS_JSON)
    print(f"  Loaded {len(papers_dict)} papers")

    global_gt_by_type = defaultdict(list)
    global_pred_by_type = defaultdict(list)

    per_file_stats = []

    hallucination_stats = {
        'total_checked': 0,
        'fully_supported': 0,
        'partially_supported': 0,
        'hallucinated': 0,
        'hallucinated_items': [],
        'partially_supported_items': []
    }

    for file_name in sorted(common_files):
        print(f"\n{'='*80}")
        print(f"Processing: {file_name}.ann")

        gt_entities = parse_ann_file(gt_files[file_name])
        pred_entities = parse_ann_file(pred_files[file_name])

        print(f"  Parsed {len(gt_entities)} GT entities, {len(pred_entities)} predicted entities")

        gt_filtered = [(t, txt, ln) for t, txt, ln in gt_entities 
                       if not should_exclude_entity(t)]
        pred_filtered = [(t, txt, ln) for t, txt, ln in pred_entities 
                         if not should_exclude_entity(t)]

        print(f"  After filtering: {len(gt_filtered)} GT entities, {len(pred_filtered)} predicted entities")

        source_sections = papers_dict.get(file_name, {})
        source_text = "\n\n".join([
            f"### {section.upper()}\n{text}"
            for section, text in source_sections.items()
            if text
        ])

        file_hallucinations = []
        file_partial_support = []

        for entity_type, pred_text, pred_line in pred_filtered:
            is_found, confidence = check_hallucination(
                pred_text, source_text, HALLUCINATION_THRESHOLD
            )

            hallucination_stats['total_checked'] += 1

            if confidence >= 0.85:
                hallucination_stats['fully_supported'] += 1
            elif confidence >= 0.70:
                hallucination_stats['partially_supported'] += 1
                item = f"{file_name} - {entity_type}: {pred_text} (confidence: {confidence:.2f})"
                hallucination_stats['partially_supported_items'].append(item)
                file_partial_support.append(item)
            else:
                hallucination_stats['hallucinated'] += 1
                item = f"{file_name} - {entity_type}: {pred_text} (confidence: {confidence:.2f})"
                hallucination_stats['hallucinated_items'].append(item)
                file_hallucinations.append(item)

        if file_hallucinations:
            print(f"   Hallucinations detected: {len(file_hallucinations)}")
        if file_partial_support:
            print(f"   Partial support detected: {len(file_partial_support)}")

        gt_normalized = [(normalize_entity_type(t), txt, ln, file_name) 
                        for t, txt, ln in gt_filtered]
        pred_normalized = [(normalize_entity_type(t), txt, ln, file_name) 
                          for t, txt, ln in pred_filtered]

        pred_anns_dict = defaultdict(list)
        gt_anns_dict = defaultdict(list)

        for entity_type, text, _, _ in pred_normalized:
            pred_anns_dict[entity_type].append(text)

        for entity_type, text, _, _ in gt_normalized:
            gt_anns_dict[entity_type].append(text)

        deepeval_scores = {}
        if USE_DEEPEVAL_LLM_JUDGE and DEEPEVAL_AVAILABLE and source_text:
            print(f"  Running DeepEval LLM-as-judge evaluation...")
            deepeval_scores = evaluate_with_deepeval_judge(
                file_name,
                dict(pred_anns_dict),
                dict(gt_anns_dict),
                source_text
            )
            if deepeval_scores:
                print(f"    Completeness: {deepeval_scores.get('completeness_score', 'N/A')}")
                print(f"    Accuracy: {deepeval_scores.get('accuracy_score', 'N/A')}")
                print(f"    Consistency: {deepeval_scores.get('consistency_score', 'N/A')}")

        file_gt_by_type = defaultdict(list)
        file_pred_by_type = defaultdict(list)

        for entity in gt_normalized:
            entity_type = entity[0]
            file_gt_by_type[entity_type].append(entity)

        for entity in pred_normalized:
            entity_type = entity[0]
            file_pred_by_type[entity_type].append(entity)

        file_metrics = calculate_metrics_per_type(file_gt_by_type, file_pred_by_type)

        if file_metrics:
            file_tp = sum(m['tp'] for m in file_metrics.values())
            file_fp = sum(m['fp'] for m in file_metrics.values())
            file_fn = sum(m['fn'] for m in file_metrics.values())

            file_precision = file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0
            file_recall = file_tp / (file_tp + file_fn) if (file_tp + file_fn) > 0 else 0
            file_f1 = 2 * file_precision * file_recall / (file_precision + file_recall) if (file_precision + file_recall) > 0 else 0

            file_stat = {
                'file_name': file_name,
                'gt_count': len(gt_normalized),
                'pred_count': len(pred_normalized),
                'tp': file_tp,
                'fp': file_fp,
                'fn': file_fn,
                'precision': file_precision,
                'recall': file_recall,
                'f1': file_f1
            }

            if deepeval_scores:
                file_stat['deepeval_completeness'] = deepeval_scores.get('completeness_score', None)
                file_stat['deepeval_accuracy'] = deepeval_scores.get('accuracy_score', None)
                file_stat['deepeval_consistency'] = deepeval_scores.get('consistency_score', None)

            per_file_stats.append(file_stat)

            print(f"  File Metrics: P={file_precision:.4f}, R={file_recall:.4f}, F1={file_f1:.4f}")

        for entity in gt_normalized:
            entity_type = entity[0]
            global_gt_by_type[entity_type].append(entity)

        for entity in pred_normalized:
            entity_type = entity[0]
            global_pred_by_type[entity_type].append(entity)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"Total files processed: {len(common_files)}")
    print(f"Total GT entity types: {len(global_gt_by_type)}")
    print(f"Total Pred entity types: {len(global_pred_by_type)}")

    print(f"\n{'='*80}")
    print("HALLUCINATION DETECTION SUMMARY")
    print(f"Total predictions checked: {hallucination_stats['total_checked']}")
    print(f"Fully supported (>=85%): {hallucination_stats['fully_supported']}")
    print(f"Partially supported (70-84%): {hallucination_stats['partially_supported']}")
    print(f"Hallucinated (<70%): {hallucination_stats['hallucinated']}")

    if hallucination_stats['total_checked'] > 0:
        hallucination_rate = hallucination_stats['hallucinated'] / hallucination_stats['total_checked']
        print(f"Hallucination rate: {hallucination_rate:.2%}")

    if USE_DEEPEVAL_LLM_JUDGE and DEEPEVAL_AVAILABLE and per_file_stats:
        completeness_scores = [s.get('deepeval_completeness') for s in per_file_stats if s.get('deepeval_completeness') is not None]
        accuracy_scores = [s.get('deepeval_accuracy') for s in per_file_stats if s.get('deepeval_accuracy') is not None]
        consistency_scores = [s.get('deepeval_consistency') for s in per_file_stats if s.get('deepeval_consistency') is not None]

        if completeness_scores or accuracy_scores or consistency_scores:
            print(f"\n{'='*80}")
            print("DEEPEVAL LLM-AS-JUDGE SUMMARY")

            if completeness_scores:
                avg_completeness = sum(completeness_scores) / len(completeness_scores)
                print(f"Average Completeness Score: {avg_completeness:.4f} ({len(completeness_scores)} papers)")
                print(f"  Min: {min(completeness_scores):.4f}, Max: {max(completeness_scores):.4f}")

            if accuracy_scores:
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                print(f"Average Accuracy Score: {avg_accuracy:.4f} ({len(accuracy_scores)} papers)")
                print(f"  Min: {min(accuracy_scores):.4f}, Max: {max(accuracy_scores):.4f}")

            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                print(f"Average Consistency Score: {avg_consistency:.4f} ({len(consistency_scores)} papers)")
                print(f"  Min: {min(consistency_scores):.4f}, Max: {max(consistency_scores):.4f}")

    if not global_gt_by_type:
        print("\n WARNING: No ground truth entities found after filtering!")
        return None

    print("\nCalculating metrics with advanced fuzzy matching...")
    print("(Allowing many-to-many matches within each file)")
    print("(Each file is evaluated against its own ground truth only)")
    metrics = calculate_metrics_per_type(global_gt_by_type, global_pred_by_type)

    return metrics, global_gt_by_type, global_pred_by_type, per_file_stats, hallucination_stats

def plot_results(metrics, output_dir, gt_by_type, pred_by_type):
    """Create visualization plots for the evaluation results."""

    if not metrics:
        print("No metrics to plot!")
        return

    entity_types = sorted(metrics.keys())
    precisions = [metrics[t]['precision'] for t in entity_types]
    recalls = [metrics[t]['recall'] for t in entity_types]
    f1_scores = [metrics[t]['f1'] for t in entity_types]

    total_tp = sum(metrics[t]['tp'] for t in entity_types)
    total_fp = sum(metrics[t]['fp'] for t in entity_types)
    total_fn = sum(metrics[t]['fn'] for t in entity_types)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    macro_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(entity_types))
    width = 0.35

    ax1.bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='#2196F3')
    ax1.bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='#4CAF50')

    ax1.set_xlabel('Entity Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Precision and Recall by Entity Type', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(entity_types, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])

    metrics_labels = ['Micro-Avg', 'Macro-Avg']
    precision_values = [overall_precision, macro_precision]
    recall_values = [overall_recall, macro_recall]

    x_overall = np.arange(len(metrics_labels))
    width_overall = 0.35

    bars1 = ax2.bar(x_overall - width_overall/2, precision_values, width_overall, 
                    label='Precision', alpha=0.8, color='#2196F3')
    bars2 = ax2.bar(x_overall + width_overall/2, recall_values, width_overall, 
                    label='Recall', alpha=0.8, color='#4CAF50')

    ax2.set_xlabel('Averaging Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_overall)
    ax2.set_xticklabels(metrics_labels, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    sorted_indices = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)
    sorted_types = [entity_types[i] for i in sorted_indices]
    sorted_recall = [recalls[i] for i in sorted_indices]

    colors = ['#4CAF50' if r >= 0.7 else '#FF9800' if r >= 0.5 else '#f44336' for r in sorted_recall]
    ax3.barh(range(len(sorted_types)), sorted_recall, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(sorted_types)))
    ax3.set_yticklabels(sorted_types, fontsize=8)
    ax3.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax3.set_title('Recall Ranking by Entity Type', fontsize=13, fontweight='bold')
    ax3.set_xlim([0, 1.1])
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()

    for i, v in enumerate(sorted_recall):
        ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    gt_counts = [metrics[t]['gt_count'] for t in entity_types]
    pred_counts = [metrics[t]['pred_count'] for t in entity_types]

    scatter = ax4.scatter(gt_counts, pred_counts, s=150, alpha=0.6, c=recalls, 
                         cmap='RdYlGn', vmin=0, vmax=1, edgecolors='black', linewidth=0.5)

    max_val = max(max(gt_counts) if gt_counts else 1, max(pred_counts) if pred_counts else 1)
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Match', linewidth=2)

    for i, et in enumerate(entity_types):
        ax4.annotate(et, (gt_counts[i], pred_counts[i]), 
                    fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')

    ax4.set_xlabel('Ground Truth Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Prediction Count', fontsize=11, fontweight='bold')
    ax4.set_title('Entity Counts: Ground Truth vs Predictions (colored by Recall)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Recall', fontsize=10)

    ax5 = fig.add_subplot(gs[2, 0])

    match_type_counts = defaultdict(int)
    for et in entity_types:
        for match in metrics[et]['matches']:
            match_type_counts[match['match_type']] += 1

    if match_type_counts:
        match_types = sorted(match_type_counts.keys())
        counts = [match_type_counts[mt] for mt in match_types]
        colors_match = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800'][:len(match_types)]

        bars = ax5.bar(match_types, counts, color=colors_match, alpha=0.7)
        ax5.set_xlabel('Match Type', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax5.set_title('Distribution of Match Types', fontsize=13, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('tight')
    ax6.axis('off')

    summary_data = [
        ['Metric', 'Micro-Avg', 'Macro-Avg'],
        ['Precision', f'{overall_precision:.4f}', f'{macro_precision:.4f}'],
        ['Recall', f'{overall_recall:.4f}', f'{macro_recall:.4f}'],
        ['', '', ''],
        ['Total GT Entities', str(total_tp + total_fn), ''],
        ['Total Pred Entities', str(total_tp + total_fp), ''],
        ['True Positives', str(total_tp), ''],
        ['False Positives', str(total_fp), ''],
        ['False Negatives', str(total_fn), ''],
        ['', '', ''],
        ['Entity Types (GT)', str(len(gt_by_type)), ''],
        ['Entity Types (Pred)', str(len(pred_by_type)), '']
    ]

    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.45, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(3):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in [1, 2, 3]:
        table[(i, 0)].set_facecolor('#E3F2FD')

    ax6.set_title('Overall Performance Summary', fontsize=13, fontweight='bold', pad=20)

    output_path = os.path.join(output_dir, 'evaluation_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Plot saved to: {output_path}")
    plt.close()

def plot_per_file_results(per_file_stats, output_dir):
    """Plot per-file evaluation results."""

    if not per_file_stats or len(per_file_stats) < 2:
        return

    sorted_stats = sorted(per_file_stats, key=lambda x: x['recall'], reverse=True)

    file_names = [stat['file_name'] for stat in sorted_stats]
    precisions = [stat['precision'] for stat in sorted_stats]
    recalls = [stat['recall'] for stat in sorted_stats]
    f1_scores = [stat['f1'] for stat in sorted_stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    x = np.arange(len(file_names))
    width = 0.35

    ax1.bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='#2196F3')
    ax1.bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='#4CAF50')

    ax1.set_xlabel('File / Paper', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Performance by File/Paper', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)

    colors = ['#4CAF50' if r >= 0.7 else '#FF9800' if r >= 0.5 else '#f44336' for r in recalls]

    ax2.barh(range(len(file_names)), recalls, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(file_names)))
    ax2.set_yticklabels(file_names, fontsize=9)
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_title('Recall by File/Paper (Ranked)', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1.1])
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    for i, v in enumerate(recalls):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_file_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Per-file plot saved to: {output_path}")
    plt.close()

def plot_hallucination_stats(hallucination_stats, output_dir):
    """Create visualization for hallucination detection statistics."""

    if not hallucination_stats or hallucination_stats['total_checked'] == 0:
        print("No hallucination stats to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    total = hallucination_stats['total_checked']
    fully_supported = hallucination_stats['fully_supported']
    partially_supported = hallucination_stats['partially_supported']
    hallucinated = hallucination_stats['hallucinated']

    ax1 = axes[0]
    sizes = [fully_supported, partially_supported, hallucinated]
    labels = [
        f'Fully Supported\n(≥85%)\n{fully_supported} items',
        f'Partially Supported\n(70-84%)\n{partially_supported} items',
        f'Hallucinated\n(<70%)\n{hallucinated} items'
    ]
    colors = ['#4CAF50', '#FFC107', '#f44336']
    explode = (0.05, 0.05, 0.1)

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, explode=explode, textprops={'fontsize': 11})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax1.set_title('Source Text Support Distribution\n(All Predictions)', 
                  fontsize=14, fontweight='bold', pad=20)

    ax2 = axes[1]
    categories = ['Fully\nSupported\n(≥85%)', 'Partially\nSupported\n(70-84%)', 'Hallucinated\n(<70%)']
    values = [fully_supported, partially_supported, hallucinated]
    percentages = [v/total*100 for v in values]

    bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Hallucination Detection Results', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    summary_text = f'Total Predictions Checked: {total}\n'
    summary_text += f'Hallucination Rate: {hallucinated/total*100:.1f}%\n'
    summary_text += f'Full Support Rate: {fully_supported/total*100:.1f}%'

    ax2.text(0.5, 0.95, summary_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hallucination_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Hallucination stats plot saved to: {output_path}")
    plt.close()

def plot_deepeval_scores(per_file_stats, output_dir):
    """Create visualization for DeepEval LLM-as-judge scores."""

    deepeval_data = []
    for stat in per_file_stats:
        if 'deepeval_completeness' in stat or 'deepeval_accuracy' in stat or 'deepeval_consistency' in stat:
            deepeval_data.append({
                'file': stat['file_name'],
                'completeness': stat.get('deepeval_completeness'),
                'accuracy': stat.get('deepeval_accuracy'),
                'consistency': stat.get('deepeval_consistency')
            })

    if not deepeval_data:
        print("No DeepEval scores to plot")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    files = [d['file'] for d in deepeval_data]
    completeness_scores = [d['completeness'] for d in deepeval_data if d['completeness'] is not None]
    accuracy_scores = [d['accuracy'] for d in deepeval_data if d['accuracy'] is not None]
    consistency_scores = [d['consistency'] for d in deepeval_data if d['consistency'] is not None]

    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(deepeval_data))
    width = 0.25

    comp_vals = [d['completeness'] if d['completeness'] is not None else 0 for d in deepeval_data]
    acc_vals = [d['accuracy'] if d['accuracy'] is not None else 0 for d in deepeval_data]
    cons_vals = [d['consistency'] if d['consistency'] is not None else 0 for d in deepeval_data]

    ax1.bar(x - width, comp_vals, width, label='Completeness', alpha=0.8, color='#2196F3')
    ax1.bar(x, acc_vals, width, label='Accuracy', alpha=0.8, color='#4CAF50')
    ax1.bar(x + width, cons_vals, width, label='Consistency', alpha=0.8, color='#FF9800')

    ax1.set_xlabel('Paper / File', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('DeepEval LLM-as-Judge Scores by Paper', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(files, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (0.7)')

    ax2 = fig.add_subplot(gs[1, 0])
    box_data = []
    labels = []

    if completeness_scores:
        box_data.append(completeness_scores)
        labels.append('Completeness')
    if accuracy_scores:
        box_data.append(accuracy_scores)
        labels.append('Accuracy')
    if consistency_scores:
        box_data.append(consistency_scores)
        labels.append('Consistency')

    if box_data:
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))

        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('DeepEval Metric Distributions', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1.1])

        for i, data in enumerate(box_data, 1):
            mean_val = np.mean(data)
            ax2.plot(i, mean_val, 'D', color='green', markersize=8, label='Mean' if i == 1 else '')

        ax2.legend(fontsize=10)

    ax3 = fig.add_subplot(gs[1, 1])
    avg_scores = []
    metric_names = []
    colors = []

    if completeness_scores:
        avg_scores.append(np.mean(completeness_scores))
        metric_names.append('Completeness')
        colors.append('#2196F3')
    if accuracy_scores:
        avg_scores.append(np.mean(accuracy_scores))
        metric_names.append('Accuracy')
        colors.append('#4CAF50')
    if consistency_scores:
        avg_scores.append(np.mean(consistency_scores))
        metric_names.append('Consistency')
        colors.append('#FF9800')

    if avg_scores:
        bars = ax3.bar(metric_names, avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Average Score', fontsize=12, fontweight='bold')
        ax3.set_title('Average Scores Across All Papers', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 1.1])
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
        ax3.legend(fontsize=10)

        for bar, val in zip(bars, avg_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4 = fig.add_subplot(gs[2, :])

    matrix_data = []
    for d in deepeval_data:
        row = []
        if d['completeness'] is not None:
            row.append(d['completeness'])
        else:
            row.append(np.nan)
        if d['accuracy'] is not None:
            row.append(d['accuracy'])
        else:
            row.append(np.nan)
        if d['consistency'] is not None:
            row.append(d['consistency'])
        else:
            row.append(np.nan)
        matrix_data.append(row)

    matrix_data = np.array(matrix_data).T

    im = ax4.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Completeness', 'Accuracy', 'Consistency'], fontsize=10)
    ax4.set_xticks(range(len(files)))
    ax4.set_xticklabels(files, rotation=45, ha='right', fontsize=9)
    ax4.set_title('DeepEval Scores Heatmap', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Score', fontsize=11)

    for i in range(matrix_data.shape[0]):
        for j in range(matrix_data.shape[1]):
            if not np.isnan(matrix_data[i, j]):
                text = ax4.text(j, i, f'{matrix_data[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'deepeval_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" DeepEval scores plot saved to: {output_path}")
    plt.close()

def plot_comprehensive_summary(metrics, per_file_stats, hallucination_stats, output_dir, gt_by_type, pred_by_type):
    """
    Create a comprehensive summary figure combining:
    1. DeepEval average scores across all papers
    2. Hallucination detection results
    3. Overall performance metrics (micro/macro precision, recall)
    4. Performance by file/paper
    """

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    total_tp = sum(metrics[t]['tp'] for t in metrics.keys())
    total_fp = sum(metrics[t]['fp'] for t in metrics.keys())
    total_fn = sum(metrics[t]['fn'] for t in metrics.keys())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)

    if per_file_stats:
        completeness_scores = [s.get('deepeval_completeness') for s in per_file_stats if s.get('deepeval_completeness') is not None]
        accuracy_scores = [s.get('deepeval_accuracy') for s in per_file_stats if s.get('deepeval_accuracy') is not None]
        consistency_scores = [s.get('deepeval_consistency') for s in per_file_stats if s.get('deepeval_consistency') is not None]

        if completeness_scores or accuracy_scores or consistency_scores:
            ax1 = fig.add_subplot(gs[0, 0])

            avg_scores = []
            metric_names = []
            colors_deepeval = []

            if completeness_scores:
                avg_scores.append(np.mean(completeness_scores))
                metric_names.append('Completeness')
                colors_deepeval.append('#2196F3')
            if accuracy_scores:
                avg_scores.append(np.mean(accuracy_scores))
                metric_names.append('Accuracy')
                colors_deepeval.append('#4CAF50')
            if consistency_scores:
                avg_scores.append(np.mean(consistency_scores))
                metric_names.append('Consistency')
                colors_deepeval.append('#FF9800')

            bars = ax1.bar(metric_names, avg_scores, color=colors_deepeval, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('Average Score', fontsize=14, fontweight='bold')
            ax1.set_title('Average Scores Across All Papers', fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylim([0, 1.1])
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Threshold')
            ax1.legend(fontsize=11)

            for bar, val in zip(bars, avg_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=13, fontweight='bold')

    if hallucination_stats and hallucination_stats['total_checked'] > 0:
        ax2 = fig.add_subplot(gs[0, 1])

        total = hallucination_stats['total_checked']
        fully_supported = hallucination_stats['fully_supported']
        partially_supported = hallucination_stats['partially_supported']
        hallucinated = hallucination_stats['hallucinated']

        categories = ['Fully\nSupported\n(≥85%)', 'Partially\nSupported\n(70-84%)', 'Hallucinated\n(<70%)']
        values = [fully_supported, partially_supported, hallucinated]
        percentages = [v/total*100 for v in values]
        colors = ['#4CAF50', '#FFC107', '#f44336']

        bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax2.set_title('Hallucination Detection Results', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        summary_text = f'Total Predictions Checked: {total}\n'
        summary_text += f'Hallucination Rate: {hallucinated/total*100:.1f}%\n'
        summary_text += f'Full Support Rate: {fully_supported/total*100:.1f}%'

        ax2.text(0.5, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax3 = fig.add_subplot(gs[1, 0])

    methods = ['Micro-Avg', 'Macro-Avg']
    precision_vals = [overall_precision, macro_precision]
    recall_vals = [overall_recall, macro_recall]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x - width/2, precision_vals, width, label='Precision', 
                    alpha=0.8, color='#2196F3', edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x + width/2, recall_vals, width, label='Recall', 
                    alpha=0.8, color='#4CAF50', edgecolor='black', linewidth=1)

    ax3.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Averaging Method', fontsize=14, fontweight='bold')
    ax3.set_title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=12)
    ax3.legend(fontsize=12)
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    if per_file_stats and len(per_file_stats) > 1:
        ax4 = fig.add_subplot(gs[1, 1])

        file_names = [s['file_name'] for s in per_file_stats]
        precisions = [s['precision'] for s in per_file_stats]
        recalls = [s['recall'] for s in per_file_stats]

        x = np.arange(len(file_names))
        width = 0.35

        bars1 = ax4.bar(x - width/2, precisions, width, label='Precision', 
                       alpha=0.8, color='#2196F3', edgecolor='black', linewidth=0.5)
        bars2 = ax4.bar(x + width/2, recalls, width, label='Recall', 
                       alpha=0.8, color='#4CAF50', edgecolor='black', linewidth=0.5)

        ax4.set_xlabel('File / Paper', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax4.set_title('Performance by File/Paper', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xticks(x)
        ax4.set_xticklabels(file_names, rotation=45, ha='right', fontsize=10)
        ax4.legend(fontsize=11, loc='lower right')
        ax4.set_ylim([0, 1.1])
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Proteomics Metadata Extraction - Evaluation Summary', 
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, 'comprehensive_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Comprehensive summary plot saved to: {output_path}")
    plt.close()

def save_detailed_results(metrics, output_dir, gt_by_type, pred_by_type, per_file_stats=None, hallucination_stats=None):
    """Save detailed results to CSV files."""

    if not metrics:
        print("No metrics to save!")
        return None, 0, 0, 0

    df_data = []
    for entity_type, metric_dict in sorted(metrics.items()):
        df_data.append({
            'Entity Type': entity_type,
            'Precision': f"{metric_dict['precision']:.4f}",
            'Recall': f"{metric_dict['recall']:.4f}",
            'F1-Score': f"{metric_dict['f1']:.4f}",
            'True Positives': metric_dict['tp'],
            'False Positives': metric_dict['fp'],
            'False Negatives': metric_dict['fn'],
            'GT Count': metric_dict['gt_count'],
            'Pred Count': metric_dict['pred_count']
        })

    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_dir, 'detailed_metrics.csv')
    df.to_csv(csv_path, index=False, escapechar='\\', doublequote=True)
    print(f" Detailed metrics saved to: {csv_path}")

    if per_file_stats:
        per_file_df = pd.DataFrame(per_file_stats)

        base_columns = ['file_name', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 
                       'precision', 'recall', 'f1']
        column_names = ['File Name', 'GT Count', 'Pred Count', 'TP', 'FP', 'FN',
                       'Precision', 'Recall', 'F1-Score']

        if 'deepeval_completeness' in per_file_df.columns:
            base_columns.append('deepeval_completeness')
            column_names.append('DeepEval Completeness')
        if 'deepeval_accuracy' in per_file_df.columns:
            base_columns.append('deepeval_accuracy')
            column_names.append('DeepEval Accuracy')
        if 'deepeval_consistency' in per_file_df.columns:
            base_columns.append('deepeval_consistency')
            column_names.append('DeepEval Consistency')

        per_file_df = per_file_df[base_columns]
        per_file_df.columns = column_names

        for col in ['Precision', 'Recall', 'F1-Score']:
            if col in per_file_df.columns:
                per_file_df[col] = per_file_df[col].apply(lambda x: f"{x:.4f}")

        for col in ['DeepEval Completeness', 'DeepEval Accuracy', 'DeepEval Consistency']:
            if col in per_file_df.columns:
                per_file_df[col] = per_file_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

        per_file_path = os.path.join(output_dir, 'per_file_results.csv')
        per_file_df.to_csv(per_file_path, index=False, escapechar='\\', doublequote=True)
        print(f" Per-file results saved to: {per_file_path}")

    total_tp = sum(metrics[t]['tp'] for t in metrics.keys())
    total_fp = sum(metrics[t]['fp'] for t in metrics.keys())
    total_fn = sum(metrics[t]['fn'] for t in metrics.keys())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)

    summary_df = pd.DataFrame([
        {'Metric': 'Micro-averaged Precision', 'Value': f"{overall_precision:.4f}"},
        {'Metric': 'Micro-averaged Recall', 'Value': f"{overall_recall:.4f}"},
        {'Metric': 'Macro-averaged Precision', 'Value': f"{macro_precision:.4f}"},
        {'Metric': 'Macro-averaged Recall', 'Value': f"{macro_recall:.4f}"},
        {'Metric': 'Total GT Entities', 'Value': str(total_tp + total_fn)},
        {'Metric': 'Total Predicted Entities', 'Value': str(total_tp + total_fp)},
        {'Metric': 'True Positives', 'Value': str(total_tp)},
        {'Metric': 'False Positives', 'Value': str(total_fp)},
        {'Metric': 'False Negatives', 'Value': str(total_fn)}
    ])

    if hallucination_stats:
        hallucination_rows = [
            {'Metric': '', 'Value': ''},
            {'Metric': '=== HALLUCINATION DETECTION ===', 'Value': ''},
            {'Metric': 'Total Predictions Checked', 'Value': str(hallucination_stats['total_checked'])},
            {'Metric': 'Fully Supported (>=85%)', 'Value': str(hallucination_stats['fully_supported'])},
            {'Metric': 'Partially Supported (70-84%)', 'Value': str(hallucination_stats['partially_supported'])},
            {'Metric': 'Hallucinated (<70%)', 'Value': str(hallucination_stats['hallucinated'])}
        ]
        if hallucination_stats['total_checked'] > 0:
            hallucination_rate = hallucination_stats['hallucinated'] / hallucination_stats['total_checked']
            hallucination_rows.append({'Metric': 'Hallucination Rate', 'Value': f"{hallucination_rate:.2%}"})

        summary_df = pd.concat([summary_df, pd.DataFrame(hallucination_rows)], ignore_index=True)

    if per_file_stats:
        completeness_scores = [s.get('deepeval_completeness') for s in per_file_stats if s.get('deepeval_completeness') is not None]
        accuracy_scores = [s.get('deepeval_accuracy') for s in per_file_stats if s.get('deepeval_accuracy') is not None]
        consistency_scores = [s.get('deepeval_consistency') for s in per_file_stats if s.get('deepeval_consistency') is not None]

        if completeness_scores or accuracy_scores or consistency_scores:
            deepeval_rows = [
                {'Metric': '', 'Value': ''},
                {'Metric': '=== DEEPEVAL LLM-AS-JUDGE ===', 'Value': ''}
            ]

            if completeness_scores:
                avg_completeness = sum(completeness_scores) / len(completeness_scores)
                deepeval_rows.extend([
                    {'Metric': 'Avg Completeness Score', 'Value': f"{avg_completeness:.4f}"},
                    {'Metric': 'Min Completeness', 'Value': f"{min(completeness_scores):.4f}"},
                    {'Metric': 'Max Completeness', 'Value': f"{max(completeness_scores):.4f}"}
                ])

            if accuracy_scores:
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                deepeval_rows.extend([
                    {'Metric': 'Avg Accuracy Score', 'Value': f"{avg_accuracy:.4f}"},
                    {'Metric': 'Min Accuracy', 'Value': f"{min(accuracy_scores):.4f}"},
                    {'Metric': 'Max Accuracy', 'Value': f"{max(accuracy_scores):.4f}"}
                ])

            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                deepeval_rows.extend([
                    {'Metric': 'Avg Consistency Score', 'Value': f"{avg_consistency:.4f}"},
                    {'Metric': 'Min Consistency', 'Value': f"{min(consistency_scores):.4f}"},
                    {'Metric': 'Max Consistency', 'Value': f"{max(consistency_scores):.4f}"}
                ])

            deepeval_rows.append({'Metric': 'Papers Evaluated with LLM Judge', 'Value': str(len(completeness_scores))})
            summary_df = pd.concat([summary_df, pd.DataFrame(deepeval_rows)], ignore_index=True)

    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False, escapechar='\\', doublequote=True)
    print(f" Summary statistics saved to: {summary_path}")

    if hallucination_stats:
        hallucination_txt_path = os.path.join(output_dir, 'hallucination_report.txt')
        with open(hallucination_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HALLUCINATION DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Predictions Checked: {hallucination_stats['total_checked']}\n")
            f.write(f"Fully Supported (≥85% confidence): {hallucination_stats['fully_supported']}\n")
            f.write(f"Partially Supported (70-84% confidence): {hallucination_stats['partially_supported']}\n")
            f.write(f"Hallucinated (<70% confidence): {hallucination_stats['hallucinated']}\n")

            if hallucination_stats['total_checked'] > 0:
                hallucination_rate = hallucination_stats['hallucinated'] / hallucination_stats['total_checked']
                partial_rate = hallucination_stats['partially_supported'] / hallucination_stats['total_checked']
                supported_rate = hallucination_stats['fully_supported'] / hallucination_stats['total_checked']

                f.write(f"\nHallucination Rate: {hallucination_rate:.2%}\n")
                f.write(f"Partial Support Rate: {partial_rate:.2%}\n")
                f.write(f"Full Support Rate: {supported_rate:.2%}\n")

            f.write("\n" + "=" * 80 + "\n\n")

            if hallucination_stats['hallucinated_items']:
                f.write(f"HALLUCINATED ITEMS ({len(hallucination_stats['hallucinated_items'])})\n")
                f.write("=" * 80 + "\n")
                f.write("These predictions have <70% confidence match with source text:\n\n")
                for i, item in enumerate(hallucination_stats['hallucinated_items'], 1):
                    f.write(f"{i}. {item}\n")
                f.write("\n" + "=" * 80 + "\n\n")

            if hallucination_stats['partially_supported_items']:
                f.write(f"PARTIALLY SUPPORTED ITEMS ({len(hallucination_stats['partially_supported_items'])})\n")
                f.write("=" * 80 + "\n")
                f.write("These predictions have 70-84% confidence match with source text:\n\n")
                for i, item in enumerate(hallucination_stats['partially_supported_items'], 1):
                    f.write(f"{i}. {item}\n")
                f.write("\n" + "=" * 80 + "\n")

        print(f" Hallucination text report saved to: {hallucination_txt_path}")

        if hallucination_stats['hallucinated_items']:
            hallucinated_df = pd.DataFrame([
                {'Item': item} for item in hallucination_stats['hallucinated_items']
            ])
            hallucinated_path = os.path.join(output_dir, 'hallucinated_items.csv')
            hallucinated_df.to_csv(hallucinated_path, index=False, escapechar='\\', doublequote=True)
            print(f" Hallucinated items CSV saved to: {hallucinated_path}")

        if hallucination_stats['partially_supported_items']:
            partial_df = pd.DataFrame([
                {'Item': item} for item in hallucination_stats['partially_supported_items']
            ])
            partial_path = os.path.join(output_dir, 'partially_supported_items.csv')
            partial_df.to_csv(partial_path, index=False, escapechar='\\', doublequote=True)
            print(f" Partially supported items CSV saved to: {partial_path}")

    if per_file_stats:
        deepeval_data = []
        deepeval_reasons = {}

        for stat in per_file_stats:
            if 'deepeval_completeness' in stat or 'deepeval_accuracy' in stat or 'deepeval_consistency' in stat:
                deepeval_data.append({
                    'File Name': stat['file_name'],
                    'Completeness Score': stat.get('deepeval_completeness', 'N/A'),
                    'Accuracy Score': stat.get('deepeval_accuracy', 'N/A'),
                    'Consistency Score': stat.get('deepeval_consistency', 'N/A')
                })

        if deepeval_data:
            deepeval_df = pd.DataFrame(deepeval_data)
            deepeval_path = os.path.join(output_dir, 'deepeval_scores.csv')
            deepeval_df.to_csv(deepeval_path, index=False, escapechar='\\', doublequote=True)
            print(f" DeepEval scores CSV saved to: {deepeval_path}")

            deepeval_txt_path = os.path.join(output_dir, 'deepeval_report.txt')
            with open(deepeval_txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DEEPEVAL LLM-AS-JUDGE EVALUATION REPORT\n")
                f.write("=" * 80 + "\n\n")

                f.write("This report uses GPT-4o as an LLM judge to evaluate proteomics metadata extraction.\n")
                f.write("The judge reads the input JSON, finds the proteomics/MS experiment description,\n")
                f.write("and evaluates the pipeline outputs.\n\n")

                f.write("METRICS:\n")
                f.write("=" * 80 + "\n\n")

                f.write("1. COMPLETENESS: Does the extraction cover all entity types?\n")
                f.write("   - Reads input JSON to find proteomics/MS experiment description\n")
                f.write("   - Checks: Are all mentioned entity types extracted?\n")
                f.write("   - Identifies missing entity types that should have been extracted\n")
                f.write("   - Focus: COVERAGE of entity types, not individual values\n")
                f.write("   - Scores: 0.9-1.0 (all types), 0.7-0.8 (most types), 0.5-0.6 (several missing), 0.0-0.4 (many gaps)\n\n")

                f.write("2. ACCURACY: Are extractions correct and from the source?\n")
                f.write("   - For each extracted value: Does it exist in the input text?\n")
                f.write("   - For each extracted value: Is it extracted with the same wording?\n")
                f.write("   - Checks for hallucinations (values not in source)\n")
                f.write("   - Checks for extraction fidelity (exact terminology preserved)\n")
                f.write("   - Focus: EXISTENCE and FIDELITY to source text\n")
                f.write("   - Scores: 0.9-1.0 (all accurate), 0.7-0.8 (minor differences), 0.5-0.6 (some errors), 0.0-0.4 (many errors)\n\n")

                f.write("3. CONSISTENCY: Does it represent a real proteomics experiment?\n")
                f.write("   - Checks: Do the extracted pieces fit together coherently?\n")
                f.write("   - Validates: Are the values scientifically realistic for proteomics?\n")
                f.write("   - Looks for: Internal contradictions or incompatibilities\n")
                f.write("   - Assesses: Whether extraction describes a viable experiment workflow\n")
                f.write("   - Focus: SCIENTIFIC COHERENCE and experimental validity\n")
                f.write("   - Scores: 0.9-1.0 (fully coherent), 0.7-0.8 (minor issues), 0.5-0.6 (some contradictions), 0.0-0.4 (incoherent)\n\n")

                f.write("=" * 80 + "\n\n")

                completeness_scores = [d['Completeness Score'] for d in deepeval_data if d['Completeness Score'] != 'N/A']
                accuracy_scores = [d['Accuracy Score'] for d in deepeval_data if d['Accuracy Score'] != 'N/A']
                consistency_scores = [d['Consistency Score'] for d in deepeval_data if d['Consistency Score'] != 'N/A']

                f.write("SUMMARY STATISTICS\n")
                f.write("=" * 80 + "\n\n")

                if completeness_scores:
                    avg_comp = sum(completeness_scores) / len(completeness_scores)
                    f.write(f"COMPLETENESS SCORES\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Average: {avg_comp:.4f}\n")
                    f.write(f"Min: {min(completeness_scores):.4f}\n")
                    f.write(f"Max: {max(completeness_scores):.4f}\n")
                    f.write(f"Papers evaluated: {len(completeness_scores)}\n\n")

                if accuracy_scores:
                    avg_acc = sum(accuracy_scores) / len(accuracy_scores)
                    f.write(f"ACCURACY SCORES\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Average: {avg_acc:.4f}\n")
                    f.write(f"Min: {min(accuracy_scores):.4f}\n")
                    f.write(f"Max: {max(accuracy_scores):.4f}\n")
                    f.write(f"Papers evaluated: {len(accuracy_scores)}\n\n")

                if consistency_scores:
                    avg_cons = sum(consistency_scores) / len(consistency_scores)
                    f.write(f"CONSISTENCY SCORES\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Average: {avg_cons:.4f}\n")
                    f.write(f"Min: {min(consistency_scores):.4f}\n")
                    f.write(f"Max: {max(consistency_scores):.4f}\n")
                    f.write(f"Papers evaluated: {len(consistency_scores)}\n\n")

                f.write("=" * 80 + "\n")
                f.write("PER-PAPER SCORES\n")
                f.write("=" * 80 + "\n\n")

                for data in deepeval_data:
                    f.write(f"File: {data['File Name']}\n")
                    f.write("-" * 80 + "\n")
                    if data['Completeness Score'] != 'N/A':
                        f.write(f"  Completeness: {data['Completeness Score']:.4f}\n")
                    if data['Accuracy Score'] != 'N/A':
                        f.write(f"  Accuracy: {data['Accuracy Score']:.4f}\n")
                    if data['Consistency Score'] != 'N/A':
                        f.write(f"  Consistency: {data['Consistency Score']:.4f}\n")
                    f.write("\n")

            print(f" DeepEval text report saved to: {deepeval_txt_path}")

    match_details_data = []
    for entity_type, metric_dict in sorted(metrics.items()):
        for match in metric_dict['matches']:
            match_details_data.append({
                'Entity_Type': entity_type,
                'GT_File': match['gt_file'],
                'GT_Text': match['gt_text'],
                'GT_Line': match['gt_line'],
                'Pred_File': match['pred_file'],
                'Pred_Text': match['pred_text'],
                'Pred_Line': match['pred_line'],
                'Similarity': f"{match['similarity']:.4f}",
                'Match_Type': match['match_type']
            })

    if match_details_data:
        match_df = pd.DataFrame(match_details_data)
        match_path = os.path.join(output_dir, 'match_details.csv')
        match_df.to_csv(match_path, index=False, escapechar='\\', doublequote=True)
        print(f" Match details saved to: {match_path}")

    fn_data = []
    for entity_type, metric_dict in sorted(metrics.items()):
        for unmatched in metric_dict['unmatched_gt']:
            fn_data.append({
                'Entity_Type': entity_type,
                'File': unmatched['gt_file'],
                'GT_Text': unmatched['gt_text'],
                'GT_Line': unmatched['gt_line']
            })

    if fn_data:
        fn_df = pd.DataFrame(fn_data)
        fn_path = os.path.join(output_dir, 'false_negatives.csv')
        fn_df.to_csv(fn_path, index=False, escapechar='\\', doublequote=True)
        print(f" False negatives saved to: {fn_path}")

    fp_data = []
    for entity_type, metric_dict in sorted(metrics.items()):
        for unmatched in metric_dict['unmatched_pred']:
            fp_data.append({
                'Entity_Type': entity_type,
                'File': unmatched['pred_file'],
                'Pred_Text': unmatched['pred_text'],
                'Pred_Line': unmatched['pred_line']
            })

    if fp_data:
        fp_df = pd.DataFrame(fp_data)
        fp_path = os.path.join(output_dir, 'false_positives.csv')
        fp_df.to_csv(fp_path, index=False, escapechar='\\', doublequote=True)
        print(f" False positives saved to: {fp_path}")

    return df, overall_f1, overall_precision, overall_recall

def print_results_table(metrics):
    """Print formatted results table to console."""

    if not metrics:
        print("\n No metrics to display!")
        return

    print("\n" + "="*100)
    print(f"{'Entity Type':<35} {'Precision':>10} {'Recall':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'GT':>6} {'Pred':>6}")

    for entity_type in sorted(metrics.keys()):
        m = metrics[entity_type]
        print(f"{entity_type:<35} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['gt_count']:>6} {m['pred_count']:>6}")

    total_tp = sum(metrics[t]['tp'] for t in metrics.keys())
    total_fp = sum(metrics[t]['fp'] for t in metrics.keys())
    total_fn = sum(metrics[t]['fn'] for t in metrics.keys())
    total_gt = sum(metrics[t]['gt_count'] for t in metrics.keys())
    total_pred = sum(metrics[t]['pred_count'] for t in metrics.keys())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)

    print(f"\n{'MICRO-AVERAGED':<35} {overall_precision:>10.4f} {overall_recall:>10.4f} "
          f"{total_tp:>6} {total_fp:>6} {total_fn:>6} {total_gt:>6} {total_pred:>6}")
    print(f"{'MACRO-AVERAGED':<35} {macro_precision:>10.4f} {macro_recall:>10.4f}")
    print("="*100 + "\n")

global_gt_by_type = None
global_pred_by_type = None

if __name__ == "__main__":
    print("NER EVALUATION SCRIPT - ADVANCED FUZZY MATCHING")
    print(f"Ground Truth Dir: {GROUND_TRUTH_DIR}")
    print(f"Predictions Dir: {PREDICTIONS_DIR}")
    print(f"Results Dir: {RESULTS_DIR}")
    print(f"\nExcluding entity types: {sorted(EXCLUDED_TYPES)}")
    print("  (Exact matches only - e.g., 'Time', 'Temperature', 'Compound', 'FactorValue[X]')")
    print("\nMatching Strategy: Many-to-Many Fuzzy Matching (File-Specific)")
    print("  - Each file is evaluated against its own ground truth only")
    print("  - Within each file: one prediction can match multiple GTs")
    print("  - Within each file: multiple predictions can match one GT")
    print("  - Handles substrings, abbreviations, and partial matches")
    print("  - Results are aggregated across all files for overall metrics")

    result = evaluate_files(GROUND_TRUTH_DIR, PREDICTIONS_DIR)

    if result is not None:
        metrics, gt_by_type, pred_by_type, per_file_stats, hallucination_stats = result
        global_gt_by_type = gt_by_type
        global_pred_by_type = pred_by_type

        if metrics:
            print_results_table(metrics)

            if per_file_stats:
                print("\n" + "="*80)
                print("PER-FILE SUMMARY")
                for stat in per_file_stats:
                    print(f"{stat['file_name']:<30} Precision={stat['precision']:.4f} Recall={stat['recall']:.4f}")

            save_detailed_results(metrics, RESULTS_DIR, gt_by_type, pred_by_type, per_file_stats, hallucination_stats)

            plot_results(metrics, RESULTS_DIR, gt_by_type, pred_by_type)

            if per_file_stats and len(per_file_stats) > 1:
                plot_per_file_results(per_file_stats, RESULTS_DIR)

            if hallucination_stats:
                plot_hallucination_stats(hallucination_stats, RESULTS_DIR)

            if per_file_stats and USE_DEEPEVAL_LLM_JUDGE and DEEPEVAL_AVAILABLE:
                plot_deepeval_scores(per_file_stats, RESULTS_DIR)

            print("\nGenerating comprehensive summary figure...")
            plot_comprehensive_summary(metrics, per_file_stats, hallucination_stats, RESULTS_DIR, gt_by_type, pred_by_type)

            print("\n" + "="*80)
            print(" EVALUATION COMPLETE!")
            print(f"\nResults saved to: {RESULTS_DIR}")
            print("\n=== CSV FILES ===")
            print("  - detailed_metrics.csv: Per-entity-type metrics")
            print("  - per_file_results.csv: Results for each file/paper (includes DeepEval scores)")
            print("  - summary_statistics.csv: Overall performance (includes hallucination & DeepEval stats)")
            print("  - match_details.csv: All matched entities with scores")
            print("  - false_negatives.csv: GT entities that were missed")
            print("  - false_positives.csv: Predicted entities with no GT match")
            print("  - hallucinated_items.csv: Predictions with low source support (<70%)")
            print("  - partially_supported_items.csv: Predictions with partial source support (70-84%)")
            print("  - deepeval_scores.csv: Per-paper LLM-as-judge scores")
            print("\n=== TEXT REPORTS ===")
            print("  - hallucination_report.txt: Detailed hallucination analysis")
            print("  - deepeval_report.txt: LLM-as-judge evaluation report")
            print("\n=== VISUALIZATIONS ===")
            print("  - comprehensive_summary.png:  ALL-IN-ONE summary (6 panels)")
            print("  - evaluation_results.png: Detailed evaluation (6 panels)")
            print("  - hallucination_stats.png: Hallucination detection visualization")
            print("  - deepeval_scores.png: LLM-as-judge scores visualization (4 panels)")
            if per_file_stats and len(per_file_stats) > 1:
                print("  - per_file_results.png: Per-file/paper performance comparison")
        else:
            print("\n No metrics calculated.")
    else:
        print("\n Evaluation failed.")