#!/usr/bin/env python3

"""
Export Mie dataset to CSV format.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def export_to_csv(npz_path: str, output_dir: str = "data/processed") -> None:
    """
    Export NPZ dataset to CSV files.
    Creates a long-format CSV with all data.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading dataset from {npz_path}...")
    data = np.load(npz_path)
    
    wavelengths = data["wavelengths_nm"]
    X = data["X"]
    
    # Get list of output arrays (all Y_* arrays)
    output_keys = [key for key in sorted(data.files) if key.startswith("Y_")]
    
    print(f"Creating long-format CSV with {X.shape[0]:,} samples × {len(wavelengths)} wavelengths...")
    
    # Create long-format dataframe
    rows = []
    
    for sample_idx in range(X.shape[0]):
        radius = X[sample_idx, 0]
        n = X[sample_idx, 1]
        k = X[sample_idx, 2]
        
        for wl_idx, wavelength in enumerate(wavelengths):
            row = {
                "sample_id": sample_idx,
                "radius_nm": radius,
                "n": n,
                "k": k,
                "wavelength_nm": wavelength,
            }
            
            # Add all output properties
            for output_key in output_keys:
                value = data[output_key][sample_idx, wl_idx]
                row[output_key] = value
            
            rows.append(row)
        
        # Progress indicator
        if (sample_idx + 1) % 10000 == 0:
            print(f"  Processed {sample_idx + 1:,} samples...")
    
    print("Creating DataFrame...")
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = output_dir / "mie_dataset_v1_long.csv"
    print(f"Saving to {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    file_size = csv_path.stat().st_size / (1024**3)
    print(f"✓ Created: {csv_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  File size: {file_size:.2f} GB")
    
    # Also create a summary CSV with just inputs
    print("\nCreating input summary CSV...")
    df_inputs = pd.DataFrame({
        "sample_id": range(X.shape[0]),
        "radius_nm": X[:, 0],
        "n": X[:, 1],
        "k": X[:, 2],
    })
    
    inputs_path = output_dir / "mie_dataset_v1_inputs.csv"
    df_inputs.to_csv(inputs_path, index=False)
    print(f"✓ Created: {inputs_path}")
    print(f"  Rows: {len(df_inputs):,}")
    
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  1. {inputs_path}")
    print(f"     → Just the input parameters (radius, n, k) for all samples")
    print(f"  2. {csv_path}")
    print(f"     → Full long-format data (all samples × wavelengths × properties)")
    print(f"\nLong format structure (first few rows):")
    print(df.head(10).to_string(index=False))
    print("\n")


if __name__ == "__main__":
    inspect_path = "data/processed/mie_dataset_v1.npz"
    if Path(inspect_path).exists():
        export_to_csv(inspect_path)
    else:
        print(f"Error: Dataset not found at {inspect_path}")
        print("Please run: python scripts/generate_dataset.py")
