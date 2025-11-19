import json
import os
from pathlib import Path
from collections import defaultdict


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_outputs(output_files):
    """Merge multiple pipeline outputs into a single structure keyed by filename."""
    merged = defaultdict(dict)
    
    for category, filepath in output_files.items():
        data = load_json_file(filepath)
        
        for entry in data:
            filename = entry['filename']
            cleaned_entry = {
                k: v for k, v in entry.items() 
                if not k.startswith('_counts') and k != 'filename'
            }
            merged[filename][category] = cleaned_entry
    
    return dict(merged)


def generate_simple_annotations(paper_data, filename):
    """
    Generate simple key-value annotations from paper metadata.
    
    Format: EntityType: value (one per line)
    """
    annotations = []
    
    # Define field mappings: EntityType -> (category, field_name)
    field_mappings = {
        # Biological
        'Organism': ('biological', 'unique_organisms'),
        'OrganismPart': ('biological', 'unique_organism_parts'),
        'Strain': ('biological', 'strains'),
        'Age': ('biological', 'ages'),
        'Sex': ('biological', 'sexes'),
        
        # Clinical/Experimental
        'Disease': ('clinical', 'unique_diseases'),
        'CellLine': ('clinical', 'unique_cell_lines'),
        'CellType': ('clinical', 'unique_cell_types'),
        'Treatment': ('clinical', 'unique_treatments'),
        'TumorStage': ('clinical', 'tumor_stages'),
        'TumorGrade': ('clinical', 'tumor_grades'),
        'TumorCellularity': ('clinical', 'tumor_cellularity'),
        'TumorSize': ('clinical', 'tumor_sizes'),
        'TumorSite': ('clinical', 'tumor_sites'),
        'SampleTreatment': ('clinical', 'sample_treatments'),
        'Staining': ('clinical', 'staining_methods'),
        
        # Data Analysis
        'SearchEngine': ('data_analysis', 'unique_search_engines'),
        'QuantificationMethod': ('data_analysis', 'unique_quant_methods'),
        'Experiment': ('data_analysis', 'unique_experiments'),
        'MissedCleavages': ('data_analysis', 'missed_cleavages'),
        'FactorValue': ('data_analysis', 'factor_values'),
        'NumberOfTechnicalReplicates': ('data_analysis', 'technical_replicates'),
        'NumberOfSamples': ('data_analysis', 'number_of_samples'),
        'BiologicalReplicate': ('data_analysis', 'biological_replicates'),
        'PooledSample': ('data_analysis', 'pooled_samples'),
        'SupplementaryFile': ('data_analysis', 'supplementary_files'),
        
        # MS Instruments
        'Instrument': ('ms_instruments', 'unique_instruments'),
        'AcquisitionMethod': ('ms_instruments', 'unique_acquisition_methods'),
        'FragmentationMethod': ('ms_instruments', 'unique_fragmentation_methods'),
        'IonizationType': ('ms_instruments', 'unique_ionization_types'),
        'MS2Analyzer': ('ms_instruments', 'unique_ms2_analyzers'),
        'TechnologyType': ('ms_instruments', 'unique_technology_types'),
        
        # Sample Prep
        'CleavageAgent': ('sample_prep', 'unique_cleavage_agents'),
        'AlkylationReagent': ('sample_prep', 'unique_alkylation_reagents'),
        'ReductionReagent': ('sample_prep', 'unique_reduction_reagents'),
        'Label': ('sample_prep', 'unique_labels'),
        'Modification': ('sample_prep', 'unique_modifications'),
        'Depletion': ('sample_prep', 'unique_depletions'),
        'Compound': ('sample_prep', 'unique_compounds'),
        
        # Separation
        'Chromatography': ('separation', 'unique_chromatography_methods'),
        'EnrichmentMethod': ('separation', 'unique_enrichment_methods'),
        'GradientTime': ('separation', 'gradient_times'),
        'FlowRate': ('separation', 'flow_rates'),
        'NumberOfFractions': ('separation', 'number_of_fractions'),
        'Separation': ('separation', 'separation_techniques'),
        'FractionationMethod': ('separation', 'fractionation_methods'),
        'Bait': ('separation', 'baits'),
        'Time': ('separation', 'times'),  
        'Temperature': ('separation', 'temperatures'),  
    }
    
    # Generate annotations for each entity type
    for entity_type, (category, field_name) in sorted(field_mappings.items()):
        category_data = paper_data.get(category, {})
        if not category_data:
            continue
        
        values = category_data.get(field_name, [])
        
        if not values:
            continue
        
        # Handle both lists and single values
        if not isinstance(values, list):
            values = [values]
        
        for value in values:
            if not value:  
                continue
            
            # Create simple annotation: EntityType: value
            annotation = f"{entity_type}: {value}"
            annotations.append(annotation)
    
    return annotations


def save_simple_ann_files(merged_data, output_dir):
    """Generate and save simple .ann files for each paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, categories in merged_data.items():
        # Generate annotations
        annotations = generate_simple_annotations(categories, filename)
        
        # Determine output filename
        base_name = Path(filename).stem
        ann_filename = f"{base_name}.ann"
        ann_path = os.path.join(output_dir, ann_filename)
        
        # Save .ann file
        with open(ann_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(annotations))
        
        print(f"Generated: {ann_filename} ({len(annotations)} annotations)")


def main():
    """Main execution function."""
    print("=" * 80)
    print("DocETL Simple Annotation Generator (FIXED)")
    print("=" * 80)
    
    input_base_path = r'F:\test_metadata\other_test\output'
    output_base_path = r'F:\\test_metadata\other_test\outputs\simple_annotations'
    
    output_files = {
        'biological': os.path.join(input_base_path, 'biological_info_complete.json'),
        'clinical': os.path.join(input_base_path, 'clinical_experimental_complete.json'),
        'data_analysis': os.path.join(input_base_path, 'data_analysis_complete.json'),
        'ms_instruments': os.path.join(input_base_path, 'ms_instruments_complete.json'),
        'sample_prep': os.path.join(input_base_path, 'sample_prep_complete.json'),
        'separation': os.path.join(input_base_path, 'separation_complete.json'),
    }
    
    print(f"\n Using input directory: {input_base_path}")
    print(f"Using output directory: {output_base_path}")
    
    # Verify input files exist
    missing_files = []
    for category, filepath in output_files.items():
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        print("\nERROR: Could not find input files!")
        for path in missing_files:
            print(f"  Missing: {path}")
        return
    
    print("\nLoading pipeline outputs...")
    for category, filepath in output_files.items():
        data = load_json_file(filepath)
        print(f"  ✓ {category}: {len(data)} papers")
    
    # Merge all outputs
    print("\n Merging outputs by filename...")
    merged_data = merge_outputs(output_files)
    print(f" Merged data for {len(merged_data)} papers")
    
    # Generate simple .ann files
    print("\n Generating simple .ann files...")
    save_simple_ann_files(merged_data, output_base_path)


    print(" COMPLETE!")
    print(f"\n Summary:")
    print(f"  Papers processed: {len(merged_data)}")
    print(f"  Output directory: {output_base_path}")
    print(f"\n Files generated:")
    for filename in sorted(merged_data.keys()):
        base_name = Path(filename).stem
        ann_path = os.path.join(output_base_path, f"{base_name}.ann")
        if os.path.exists(ann_path):
            with open(ann_path, encoding='utf-8') as f:
                line_count = len(f.readlines())
            print(f"  - {base_name}.ann ({line_count} annotations)")
    
    # Show sample
    if merged_data:
        sample_filename = sorted(merged_data.keys())[0]
        base_name = Path(sample_filename).stem
        ann_path = os.path.join(output_base_path, f"{base_name}.ann")
        if os.path.exists(ann_path):
            print(f"\n Sample annotations from {base_name}.ann:")
            with open(ann_path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 10:  
                        print(f"  {line.rstrip()}")
                    else:
                        break
    
    print(f"\n Done! Check the {output_base_path} directory.")


if __name__ == '__main__':
    main()