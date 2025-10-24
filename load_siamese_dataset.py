#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孪生网络数据集加载和使用示例
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class SiameseDataset(Dataset):
    """
    孪生网络数据集类
    """
    def __init__(self, csv_file, npy_file):
        """
        Args:
            csv_file: 配对信息CSV文件路径
            npy_file: 拼接指纹NPY文件路径
        """
        self.df = pd.read_csv(csv_file)
        self.fingerprints = np.load(npy_file)
        
        # 确保数据对齐
        assert len(self.df) == len(self.fingerprints), \
            f"CSV行数({len(self.df)})与指纹数量({len(self.fingerprints)})不匹配"
        
        print(f"加载数据集: {len(self.df)} 个配对")
        print(f"指纹维度: {self.fingerprints.shape}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            dict: {
                'fingerprint': torch.tensor, shape (3*fingerprint_dim,)
                'label_diff': torch.tensor, shape (1,)
                'pair_info': dict  # 其他配对信息
            }
        """
        row = self.df.iloc[idx]
        fingerprint = self.fingerprints[idx]
        
        return {
            'fingerprint': torch.tensor(fingerprint, dtype=torch.float32),
            'label_diff': torch.tensor([row['label_diff']], dtype=torch.float32),
            'pair_info': {
                'pair_index': row['pair_index'],
                'uniprot_id': row['uniprot_id'],
                'smiles': row['smiles'],
                'mutation_1_id': row['mutation_1_id'],
                'mutation_2_id': row['mutation_2_id'],
                'label_1': row['label_1'],
                'label_2': row['label_2']
            }
        }
    
    def get_fingerprint_split_dims(self):
        """
        获取指纹分割维度信息
        
        Returns:
            int: 单个指纹的维度（总维度/3）
        """
        total_dim = self.fingerprints.shape[1]
        single_fp_dim = total_dim // 3
        return single_fp_dim
    
    def split_fingerprint(self, combined_fingerprint):
        """
        将拼接的指纹分解为三部分
        
        Args:
            combined_fingerprint: 拼接的指纹，shape (3*dim,)
        
        Returns:
            tuple: (fp1, fp2, fp_diff)
        """
        single_dim = self.get_fingerprint_split_dims()
        fp1 = combined_fingerprint[:single_dim]
        fp2 = combined_fingerprint[single_dim:2*single_dim]
        fp_diff = combined_fingerprint[2*single_dim:]
        return fp1, fp2, fp_diff


def load_dataset(data_dir, batch_size=32, shuffle=True, split_ratio=0.8):
    """
    加载孪生数据集并创建训练/验证DataLoader
    
    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        shuffle: 是否打乱
        split_ratio: 训练集比例
    
    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    csv_file = os.path.join(data_dir, 'siamese_pairs.csv')
    npy_file = os.path.join(data_dir, 'siamese_fingerprints.npy')
    
    dataset = SiameseDataset(csv_file, npy_file)
    
    # 划分训练/验证集
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset


def main():
    """
    使用示例
    """
    # 假设数据在siamese_dataset目录下
    data_dir = 'siamese_dataset'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请先运行build_siamese_dataset.py生成数据集")
        return
    
    try:
        # 加载数据集
        train_loader, val_loader, dataset = load_dataset(data_dir, batch_size=16)
        
        # 显示数据集信息
        print(f"\n=== 数据集信息 ===")
        print(f"单个指纹维度: {dataset.get_fingerprint_split_dims()}")
        print(f"拼接指纹维度: {dataset.fingerprints.shape[1]}")
        
        # 示例：遍历一个batch
        print(f"\n=== 数据样例 ===")
        for batch_idx, batch in enumerate(train_loader):
            fingerprints = batch['fingerprint']  # shape: (batch_size, 3*fingerprint_dim)
            labels = batch['label_diff']          # shape: (batch_size, 1)
            
            print(f"Batch {batch_idx}:")
            print(f"  指纹形状: {fingerprints.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  标签范围: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
            
            # 展示如何分解指纹
            sample_fp = fingerprints[0]
            fp1, fp2, fp_diff = dataset.split_fingerprint(sample_fp)
            print(f"  分解指纹形状: {fp1.shape}, {fp2.shape}, {fp_diff.shape}")
            
            if batch_idx >= 2:  # 只看前3个batch
                break
                
        print("\n数据集加载成功！")
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")


if __name__ == '__main__':
    main() 