#!/usr/bin/env python3

"""
Nicely inspect and summarize the Mie dataset.
"""

import numpy as np
from pathlib import Path

def inspect_dataset(npz_path: str) -> None:
    """Load and display dataset information in a nice format."""
    
    data = np.load(npz_path)
    
    print("\n" + "="*80)
    print("MIE DATASET INSPECTION")
    print("="*80 + "\n")
    
    # Summary statistics
    print("DATASET SUMMARY:")
    print("-" * 80)
    
    keys = sorted(data.files)
    print(f"Total arrays in dataset: {len(keys)}\n")
    
    # Organize by category
    categories = {
        "Metadata": ["wavelengths_nm"],
        "Input": ["X"],
        "Efficiency Factors": ["Y_qext", "Y_qsca", "Y_qabs", "Y_qback", "Y_g"],
        "Cross-Sections": ["Y_sigma_ext", "Y_sigma_sca", "Y_sigma_abs"],
        "Normalized (log)": ["Y_log_sigma_ext_over_geo", "Y_log_sigma_sca_over_geo", "Y_log_sigma_abs_over_geo"],
    }
    
    for category, array_names in categories.items():
        print(f"\n{category}:")
        for name in array_names:
            if name in data.files:
                arr = data[name]
                print(f"  {name:<30} | Shape: {str(arr.shape):<20} | dtype: {arr.dtype}")
    
    # Detailed input information
    print("\n" + "="*80)
    print("INPUT SPACE (X)")
    print("="*80)
    
    X = data["X"]
    print(f"\nShape: {X.shape}")
    print(f"Total samples: {X.shape[0]:,}")
    print(f"Parameters per sample: {X.shape[1]}")
    
    print("\nParameter ranges:")
    print(f"  Radius (nm)         : {X[:, 0].min():.1f} - {X[:, 0].max():.1f} nm")
    print(f"  Refractive Index (n): {X[:, 1].min():.2f} - {X[:, 1].max():.2f}")
    print(f"  Absorption Coeff (k): {X[:, 2].min():.3f} - {X[:, 2].max():.3f}")
    
    # Wavelength information
    print("\n" + "="*80)
    print("WAVELENGTH COVERAGE")
    print("="*80)
    
    wavelengths = data["wavelengths_nm"]
    print(f"\nShape: {wavelengths.shape}")
    print(f"Range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"Spacing: {np.diff(wavelengths)[0]:.2f} nm (evenly spaced)")
    
    # Output statistics
    print("\n" + "="*80)
    print("OUTPUT PROPERTIES - SAMPLE STATISTICS")
    print("="*80)
    
    output_arrays = {
        "Scattering Efficiency": "Y_qsca",
        "Absorption Efficiency": "Y_qabs",
        "Extinction Efficiency": "Y_qext",
        "Scattering Cross-Section": "Y_sigma_sca",
        "Absorption Cross-Section": "Y_sigma_abs",
        "Log Scattering/Geo": "Y_log_sigma_sca_over_geo",
    }
    
    for label, array_name in output_arrays.items():
        if array_name in data.files:
            arr = data[array_name]
            print(f"\n{label}:")
            print(f"  Min:    {arr.min():12.6f}")
            print(f"  Max:    {arr.max():12.6f}")
            print(f"  Mean:   {arr.mean():12.6f}")
            print(f"  Median: {np.median(arr):12.6f}")
    
    # Show example data
    print("\n" + "="*80)
    print("EXAMPLE: First sample (index 0)")
    print("="*80)
    
    print(f"\nInput parameters:")
    print(f"  Radius: {X[0, 0]:.2f} nm")
    print(f"  n: {X[0, 1]:.2f}")
    print(f"  k: {X[0, 2]:.4f}")
    
    print(f"\nScattering spectrum (Y_qsca) at selected wavelengths:")
    qsca = data["Y_qsca"][0]
    indices = [0, len(wavelengths)//4, len(wavelengths)//2, 3*len(wavelengths)//4, -1]
    for idx in indices:
        wl = wavelengths[idx]
        val = qsca[idx]
        print(f"  {wl:6.1f} nm: {val:.6f}")
    
    print("\n" + "="*80)
    print(f"File: {npz_path}")
    print(f"File size: {Path(npz_path).stat().st_size / (1024**2):.2f} MB")
    print("="*80 + "\n")


if __name__ == "__main__":
    inspect_dataset("data/processed/mie_dataset_v1.npz")
