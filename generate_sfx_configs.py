#!/usr/bin/env python3
"""
Script to generate *_mem_sfx.yaml files from *_mem.yaml files.

This script:
1. Finds all *_mem.yaml files in the configs/combo directory
2. Extracts finetuned_model_path and determines trainable modules based on path patterns
3. Creates corresponding *_mem_sfx.yaml files with updated trainable_modules
"""

import os
import glob
from pathlib import Path
import yaml
from typing import List, Dict, Any


def determine_trainable_modules(model_path: str) -> List[str]:
    """
    Determine trainable modules based on model path patterns.
    
    Args:
        model_path: Path to the finetuned model
        
    Returns:
        List of trainable module names
    """
    if "all" in model_path:
        return [
            "prompt_encoder", 
            "image_encoder", 
            "mask_decoder", 
            "memory_encoder", 
            "memory_attention"
        ]
    elif "pe" in model_path:
        return [
            "prompt_encoder", 
            "mask_decoder", 
            "memory_encoder", 
            "memory_attention"
        ]
    else:
        return [
            "mask_decoder", 
            "memory_encoder", 
            "memory_attention"
        ]


def process_yaml_file(input_path: str, output_path: str) -> None:
    """
    Process a single YAML file and create the corresponding _sfx version.
    
    Args:
        input_path: Path to input *_mem.yaml file
        output_path: Path to output *_mem_sfx.yaml file
    """
    try:
        # Read the original YAML file
        with open(input_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract finetuned_model_path
        if 'model' not in data or 'fintuned_model_path' not in data['model']:
            print(f"Warning: No finetuned_model_path found in {input_path}")
            return
            
        model_path = data['model']['fintuned_model_path']
        
        # Determine new trainable modules
        new_trainable_modules = determine_trainable_modules(model_path)
        
        # Update trainable_modules
        data['model']['trainable_modules'] = new_trainable_modules
        
        # Write the new YAML file
        with open(output_path, 'w') as f:
            # Write the comment at the top
            f.write("# @package _global_\n\n")
            # Write the YAML content
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"Created: {output_path}")
        print(f"  Model path: {model_path}")
        print(f"  Trainable modules: {new_trainable_modules}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def main():
    """Main function to process all *_mem.yaml files."""
    # Find all *_mem.yaml files in configs/combo directory
    pattern = "configs/combo/**/*_mem.yaml"
    mem_files = glob.glob(pattern, recursive=True)
    
    if not mem_files:
        print("No *_mem.yaml files found in configs/combo directory")
        return
    
    print(f"Found {len(mem_files)} *_mem.yaml files")
    print()
    
    # Process each file
    for input_path in sorted(mem_files):
        # Generate output path by replacing _mem.yaml with _mem_sfx.yaml
        output_path = input_path.replace('_mem.yaml', '_mem_sfx.yaml')
        
        print(f"Processing: {input_path}")
        process_yaml_file(input_path, output_path)
        print()


if __name__ == "__main__":
    main()