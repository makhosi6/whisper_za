#!/usr/bin/env python3
"""
split_jsonl_by_language.py
Split JSONL content by language with 70/30 train/val split
"""

import json
import os
import random
from pathlib import Path
import sys
from collections import defaultdict

def process_jsonl(input_file="master.jsonl", train_split=0.7):
    """Process the JSONL file and split by language"""
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    # Read and group by language
    language_data = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                
                # Get language - prefer language property, fallback to audio path
                lang = record.get('language')
                if not lang and 'audio' in record:
                    # Extract language from audio path (first segment)
                    lang = record['audio'].split('/')[0]
                
                if not lang:
                    print(f"Warning: Could not determine language for line {line_num}")
                    continue
                    
                language_data[lang].append(record)
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error on line {line_num}: {e}")
    
    # Create metadata structure
    metadata = {}
    
    # Process each language
    for lang, records in language_data.items():
        print(f"\nProcessing language: {lang} ({len(records)} records)")
        
        # Create language directory
        lang_dir = Path(lang)
        lang_dir.mkdir(exist_ok=True)
        
        # Shuffle records
        random.shuffle(records)
        
        # Calculate split
        total = len(records)
        train_count = int(total * train_split)
        val_count = total - train_count
        
        print(f"  Splitting: {train_count} train ({train_split*100:.0f}%), "
              f"{val_count} validation ({(1-train_split)*100:.0f}%)")
        
        # Write master file (all records)
        master_file = lang_dir / "master.jsonl"
        with open(master_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        # Write train split
        train_file = lang_dir / "data.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for record in records[:train_count]:
                f.write(json.dumps(record) + '\n')
        
        # Write validation split
        val_file = lang_dir / "val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for record in records[train_count:]:
                f.write(json.dumps(record) + '\n')
        
        # Add to metadata
        metadata[lang] = {
            "total_records": total,
            "master": f"{lang}/master.jsonl",
            "train": f"{lang}/data.jsonl",
            "train_records": train_count,
            "validation": f"{lang}/val.jsonl",
            "validation_records": val_count,
            "split_ratio": f"{train_split:.1f}/{1-train_split:.1f}"
        }
    
    # Write metadata
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*50}")
    print("Processing Complete!")
    print(f"{'='*50}")
    
    total_all = sum(len(records) for records in language_data.values())
    print(f"Total records processed: {total_all}")
    print(f"Languages found: {', '.join(language_data.keys())}")
    print(f"\nMetadata written to: metadata.json")
    
    for lang in language_data.keys():
        print(f"\n{lang}/")
        print(f"  master.jsonl: {len(language_data[lang])} records")
        print(f"  data.jsonl: {metadata[lang]['train_records']} records")
        print(f"  val.jsonl: {metadata[lang]['validation_records']} records")

if __name__ == "__main__":
    # Use command line argument or default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "master.jsonl"
    process_jsonl(input_file)