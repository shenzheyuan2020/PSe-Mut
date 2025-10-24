#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siamese network dataset construction script
Used to build paired datasets for protein mutation effect prediction
"""

import pandas as pd
import numpy as np
import os
from itertools import combinations
from tqdm import tqdm
import argparse


def load_fingerprints(result_path, pair_ids):
    """
    Load fingerprint files
    
    Args:
        result_path: PSICHIC result path
        pair_ids: List of pair IDs to load
    
    Returns:
        dict: {pair_id: fingerprint_array}
    """
    fingerprints = {}
    interpretation_path = os.path.join(result_path, "interpretation_result")
    
    for pair_id in tqdm(pair_ids, desc="Loading fingerprints"):
        fingerprint_file = os.path.join(interpretation_path, pair_id, "fingerprint.npy")
        if os.path.exists(fingerprint_file):
            try:
                fingerprints[pair_id] = np.load(fingerprint_file)
            except Exception as e:
                print(f"Error loading {fingerprint_file}: {e}")
        else:
            print(f"Warning: Fingerprint file not found for {pair_id}")
    
    return fingerprints


def create_siamese_pairs(df, fingerprints):
    """
    Create paired data for siamese network
    
    Args:
        df: Original data DataFrame
        fingerprints: Fingerprint data dictionary
    
    Returns:
        tuple: (siamese_df, combined_fingerprints)
    """
    # Group by UNIPROT_ID and SMILES
    groups = df.groupby(['UNIPROT_ID', 'SMILES'])
    
    siamese_data = []
    combined_fingerprints = []
    
    for (uniprot_id, smiles), group in tqdm(groups, desc="Creating siamese pairs"):
        if len(group) < 2:
            continue  # Need at least 2 mutations to pair
        
        # Get all possible pairing combinations within the group
        for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
            pair_id_1 = row1['ID']
            pair_id_2 = row2['ID']
            
            # Check if corresponding fingerprint files exist
            if pair_id_1 not in fingerprints or pair_id_2 not in fingerprints:
                continue
                
            # Get fingerprints
            fingerprint_1 = fingerprints[pair_id_1]
            fingerprint_2 = fingerprints[pair_id_2]
            
            # Calculate fingerprint difference
            fingerprint_diff = fingerprint_1 - fingerprint_2
            
            # Concatenate three fingerprints: [fp1, fp2, diff]
            combined_fingerprint = np.concatenate([fingerprint_1, fingerprint_2, fingerprint_diff])
            
            # Calculate label difference
            label_diff = row1['LABEL'] - row2['LABEL']
            
            # Record pairing information
            siamese_record = {
                'pair_index': len(siamese_data),
                'uniprot_id': uniprot_id,
                'smiles': smiles,
                'mutation_1_id': pair_id_1,
                'mutation_2_id': pair_id_2,
                'protein_1': row1['Protein'],
                'protein_2': row2['Protein'],
                'ligand': row1['Ligand'],  # Ligand should be the same within the group
                'label_1': row1['LABEL'],
                'label_2': row2['LABEL'],
                'label_diff': label_diff,
                'fingerprint_shape': combined_fingerprint.shape[0]
            }
            
            siamese_data.append(siamese_record)
            combined_fingerprints.append(combined_fingerprint)
    
    siamese_df = pd.DataFrame(siamese_data)
    combined_fingerprints = np.array(combined_fingerprints)
    
    return siamese_df, combined_fingerprints


def prepare_dataframe(df):
    """
    Prepare DataFrame, ensure necessary columns exist
    
    Args:
        df: Original DataFrame
        
    Returns:
        df: Processed DataFrame
    """
    # If no ID column, auto-generate
    if 'ID' not in df.columns:
        print("No ID column found, auto-generating ID column...")
        df['ID'] = 'PAIR_' + df.index.astype(str)
        print(f"Generated {len(df)} IDs: {df['ID'].iloc[:5].tolist()} ...")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Build siamese network dataset')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Original CSV file path')
    parser.add_argument('--result_path', type=str, required=True,
                       help='PSICHIC result directory path')
    parser.add_argument('--output_dir', type=str, default=os.path.join('data','processed','siamese'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Siamese Network Dataset Construction ===")
    
    # 1. Load original data
    print(f"Loading CSV file: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    print("Original CSV file column names:")
    print(df.columns.tolist())
    print(f"Data shape: {df.shape}")
    
    # Prepare DataFrame (if ID column needs to be generated)
    df = prepare_dataframe(df)
    
    # Check necessary columns
    required_columns = ['UNIPROT_ID', 'SMILES', 'Protein', 'Ligand', 'LABEL', 'ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print("\nAvailable column names:")
        print(df.columns.tolist())
        raise ValueError(f"CSV file missing necessary columns: {missing_columns}")
    
    print(f"Original data row count: {len(df)}")
    print(f"Unique protein-molecule combination count: {df.groupby(['UNIPROT_ID', 'SMILES']).ngroups}")
    
    # 2. Load fingerprint data
    print("Loading fingerprint data...")
    pair_ids = df['ID'].tolist()
    fingerprints = load_fingerprints(args.result_path, pair_ids)
    print(f"Successfully loaded fingerprint count: {len(fingerprints)}")
    
    if len(fingerprints) == 0:
        print("\nError: No fingerprint files found!")
        print("Please check:")
        print(f"1. Is PSICHIC result path correct: {args.result_path}")
        print(f"2. Does interpretation_result directory exist")
        print(f"3. Do ID column values match PSICHIC output folder names")
        print(f"Expected fingerprint file path example: {os.path.join(args.result_path, 'interpretation_result', pair_ids[0], 'fingerprint.npy')}")
        return
    
    # 3. Create siamese pairs
    print("Creating siamese pairs...")
    siamese_df, combined_fingerprints = create_siamese_pairs(df, fingerprints)
    
    print(f"Generated pair count: {len(siamese_df)}")
    print(f"Fingerprint feature dimension: {combined_fingerprints.shape}")
    
    # 4. Save results
    csv_output = os.path.join(args.output_dir, 'siamese_pairs.csv')
    npy_output = os.path.join(args.output_dir, 'siamese_fingerprints.npy')
    
    print(f"Saving CSV file: {csv_output}")
    siamese_df.to_csv(csv_output, index=False)
    
    print(f"Saving fingerprint file: {npy_output}")
    np.save(npy_output, combined_fingerprints)
    
    # 5. Generate dataset statistics report
    stats_output = os.path.join(args.output_dir, 'dataset_stats.txt')
    with open(stats_output, 'w', encoding='utf-8') as f:
        f.write("=== Siamese Network Dataset Statistics ===\n")
        f.write(f"Original data row count: {len(df)}\n")
        f.write(f"Generated pair count: {len(siamese_df)}\n")
        f.write(f"Fingerprint feature dimension: {combined_fingerprints.shape}\n")
        f.write(f"Label difference range: [{siamese_df['label_diff'].min():.4f}, {siamese_df['label_diff'].max():.4f}]\n")
        f.write(f"Label difference mean: {siamese_df['label_diff'].mean():.4f}\n")
        f.write(f"Label difference std: {siamese_df['label_diff'].std():.4f}\n")
        f.write(f"Unique protein-molecule combination count: {siamese_df.groupby(['uniprot_id', 'smiles']).ngroups}\n")
    
    print(f"Statistics report: {stats_output}")
    print("Dataset construction completed!")


if __name__ == '__main__':
    main() 