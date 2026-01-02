#!/bin/bash

# Script: split_jsonl_by_language.sh
# Description: Split JSONL content by language, create train/val splits (70/30),
#              and generate metadata with record counts

set -e  # Exit on error

# Configuration
INPUT_FILE=${1:-"master.jsonl"}
TRAIN_SPLIT=0.7
VAL_SPLIT=0.3

# Colors for output (optional)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Processing JSONL file: $INPUT_FILE${NC}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found!${NC}"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Function to count total lines in a file
count_lines() {
    wc -l < "$1" | tr -d ' '
}

# Function to shuffle lines (if shuf is available)
shuffle_file() {
    local input_file="$1"
    if command -v shuf >/dev/null 2>&1; then
        shuf "$input_file" > "$input_file.shuffled"
        mv "$input_file.shuffled" "$input_file"
    else
        echo -e "${YELLOW}Warning: 'shuf' not found. Using sort for shuffling.${NC}"
        sort -R "$input_file" > "$input_file.shuffled" 2>/dev/null || \
        echo -e "${YELLOW}Could not shuffle file. Splitting sequentially.${NC}"
        [ -f "$input_file.shuffled" ] && mv "$input_file.shuffled" "$input_file"
    fi
}

# Extract unique languages and group by language
echo "Extracting languages and grouping data..."

# Method 1: Use language property from JSON, fallback to audio path
while IFS= read -r line; do
    if [ -n "$line" ]; then
        # Try to extract language from JSON property
        lang=$(echo "$line" | grep -o '"language":[[:space:]]*"[^"]*"' | cut -d'"' -f4)
        
        # If language property not found, extract from audio path
        if [ -z "$lang" ]; then
            lang=$(echo "$line" | grep -o '"audio":[[:space:]]*"[^"]*"' | cut -d'"' -f4 | cut -d'/' -f1)
        fi
        
        if [ -n "$lang" ]; then
            echo "$line" >> "$TEMP_DIR/$lang.jsonl.tmp"
        else
            echo -e "${YELLOW}Warning: Could not extract language from line:${NC}"
            echo "$line"
        fi
    fi
done < "$INPUT_FILE"

# Initialize metadata JSON
METADATA="metadata.json"
echo "{" > "$METADATA"
first_lang=true

# Process each language
for lang_file in "$TEMP_DIR"/*.jsonl.tmp; do
    [ -f "$lang_file" ] || continue
    
    lang=$(basename "$lang_file" .jsonl.tmp)
    echo -e "\n${GREEN}Processing language: $lang${NC}"
    
    # Create language directory
    mkdir -p "$lang"
    
    # Count total records
    total_records=$(count_lines "$lang_file")
    echo "  Total records: $total_records"
    
    # Shuffle the data
    echo "  Shuffling data..."
    shuffle_file "$lang_file"
    
    # Calculate split points
    train_count=$(echo "scale=0; $total_records * $TRAIN_SPLIT / 1" | bc)
    val_count=$((total_records - train_count))
    
    echo "  Splitting: $train_count for train (70%), $val_count for validation (30%)"
    
    # Create master file (all data for this language)
    master_file="$lang/master.jsonl"
    cp "$lang_file" "$master_file"
    
    # Create train split (first 70%)
    train_file="$lang/data.jsonl"
    head -n "$train_count" "$lang_file" > "$train_file"
    
    # Create validation split (last 30%)
    val_file="$lang/val.jsonl"
    tail -n "$val_count" "$lang_file" > "$val_file"
    
    # Verify counts
    actual_train=$(count_lines "$train_file")
    actual_val=$(count_lines "$val_file")
    
    echo "  Actual split - Train: $actual_train, Validation: $actual_val"
    
    # Add to metadata
    if [ "$first_lang" = true ]; then
        first_lang=false
    else
        echo "," >> "$METADATA"
    fi
    
    cat >> "$METADATA" <<EOF
  "$lang": {
    "total_records": $total_records,
    "master": "$lang/master.jsonl",
    "train": "$lang/data.jsonl",
    "train_records": $actual_train,
    "validation": "$lang/val.jsonl",
    "validation_records": $actual_val,
    "split_ratio": "$TRAIN_SPLIT/$VAL_SPLIT"
  }
EOF
    
    # Clean up temp file
    rm -f "$lang_file"
done

# Complete metadata JSON
echo "}" >> "$METADATA"

# Summary
echo -e "\n${GREEN}=== Processing Complete ===${NC}"
echo "Generated files:"
echo "- metadata.json (overall statistics)"

for lang_dir in */; do
    if [ -d "$lang_dir" ]; then
        lang=$(basename "$lang_dir")
        master_count=$(count_lines "$lang_dir/master.jsonl" 2>/dev/null || echo 0)
        train_count=$(count_lines "$lang_dir/data.jsonl" 2>/dev/null || echo 0)
        val_count=$(count_lines "$lang_dir/val.jsonl" 2>/dev/null || echo 0)
        
        echo "- $lang_dir"
        echo "    master.jsonl: $master_count records"
        echo "    data.jsonl (train): $train_count records"
        echo "    val.jsonl: $val_count records"
    fi
done

# Display metadata
echo -e "\n${GREEN}Metadata Preview:${NC}"
cat "$METADATA" | python -m json.tool 2>/dev/null || cat "$METADATA"

# Cleanup
rm -rf "$TEMP_DIR"
echo -e "\n${GREEN}Done!${NC}"