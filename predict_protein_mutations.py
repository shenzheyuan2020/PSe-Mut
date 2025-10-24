#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical mutation effect prediction script (FIXED VERSION - Uses Siamese Tower)
Given protein sequences (before and after mutation) and molecule list, predict mutation effects on molecular binding

Usage workflow:
1. Input protein sequences before and after mutation
2. Input molecule list to predict (CSV format)
3. Automatically generate PSICHIC fingerprints
4. Use trained Siamese Tower models for prediction
5. Output prediction results and analysis

修复说明：
- 现在使用论文最终版本的三分支孪生塔预测器（dl_trainers/predict_siamese_exchange_ensemble.py）
- 支持输出 prob_pos（突变有利概率）
- 与论文结果完全一致
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import warnings
import torch
warnings.filterwarnings('ignore')

# 导入孪生塔预测模块
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CUR_DIR)
sys.path.insert(0, os.path.join(_CUR_DIR, 'dl_trainers'))

from dl_trainers.predict_siamese_exchange_ensemble import (
    screen_and_collect_fingerprints,
    load_fp_from_existing,
    ensemble_predict
)

class ProteinMutationPredictor:
    """蛋白质突变效应预测器（基于三分支孪生塔）"""
    
    def __init__(self, model_dir="weight", psichic_weights_dir=None, psichic_model_name='Trained on PDBBind v2020'):
        self.model_dir = model_dir
        self.psichic_weights_dir = psichic_weights_dir or os.path.join('artifacts', 'weights')
        self.psichic_model_name = psichic_model_name
        self.temp_dir = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 验证模型目录
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        
        # 检查是否有 seed_* 子目录
        seed_dirs = [d for d in os.listdir(self.model_dir) 
                    if d.startswith('seed_') and os.path.isdir(os.path.join(self.model_dir, d))]
        if not seed_dirs:
            raise FileNotFoundError(f"在 {self.model_dir} 下未找到 seed_* 目录，请确认模型已训练")
        
        print(f"✅ 找到 {len(seed_dirs)} 个随机种子模型: {seed_dirs}")
        
    def validate_protein_sequence(self, sequence):
        """验证蛋白质序列"""
        # 去除空格和换行符
        sequence = sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()
        
        # 检查是否包含非标准氨基酸
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence_aa = set(sequence)
        
        if not sequence_aa.issubset(standard_aa):
            invalid_aa = sequence_aa - standard_aa
            print(f"⚠️ 警告: 发现非标准氨基酸: {invalid_aa}")
            print("自动去除非标准氨基酸...")
            sequence = ''.join([aa for aa in sequence if aa in standard_aa])
        
        if len(sequence) < 50:
            print(f"⚠️ 警告: 蛋白序列较短 ({len(sequence)} 个氨基酸)")
        
        print(f"✅ 蛋白序列验证通过: {len(sequence)} 个氨基酸")
        return sequence
    
    def validate_molecules(self, molecules_df):
        """验证分子数据"""
        if 'smiles' not in molecules_df.columns and 'Ligand' not in molecules_df.columns:
            raise ValueError("CSV文件必须包含 'smiles' 或 'Ligand' 列")
        
        # 统一列名
        if 'smiles' in molecules_df.columns and 'Ligand' not in molecules_df.columns:
            molecules_df['Ligand'] = molecules_df['smiles']
        elif 'Ligand' in molecules_df.columns and 'smiles' not in molecules_df.columns:
            molecules_df['smiles'] = molecules_df['Ligand']
        
        # 去除空值
        molecules_df = molecules_df.dropna(subset=['Ligand'])
        
        # 添加ID列（如果不存在）
        if 'id' not in molecules_df.columns and 'ID' not in molecules_df.columns:
            molecules_df['ID'] = [f"mol_{i+1}" for i in range(len(molecules_df))]
        elif 'id' in molecules_df.columns and 'ID' not in molecules_df.columns:
            molecules_df['ID'] = molecules_df['id']
        
        print(f"✅ 分子数据验证通过: {len(molecules_df)} 个分子")
        return molecules_df
    
    def create_temp_workspace(self):
        """创建临时工作空间"""
        self.temp_dir = tempfile.mkdtemp(prefix="mutation_pred_")
        print(f"📁 创建临时工作空间: {self.temp_dir}")
        return self.temp_dir
    
    def generate_psichic_fingerprints(self, protein_seq, molecules_df, output_dir, prefix=""):
        """使用 PSICHIC 生成交互指纹"""
        print(f"🔬 生成 {prefix} PSICHIC 指纹...")
        
        # 配置 PSICHIC 全局参数
        screen_and_collect_fingerprints._psichic_root = self.psichic_weights_dir
        screen_and_collect_fingerprints._psichic_model = self.psichic_model_name
        
        # 提取配体列表
        ligands = molecules_df['Ligand'].astype(str).tolist()
        
        # 生成指纹
        result_df, fp_dict = screen_and_collect_fingerprints(
            ligands=ligands,
            protein_seq=protein_seq,
            device=self.device,
            result_dir=output_dir,
            batch_size=16,
            save_interpret=True
        )
        
        print(f"✅ 成功生成 {len(fp_dict)} 个指纹")
        return result_df, fp_dict
    
    def prepare_fingerprints_for_prediction(self, wt_fp_dict, mut_fp_dict, molecules_df):
        """准备用于预测的指纹数组"""
        print("📊 准备预测数据...")
        
        # 将字典转为数组（按照分子顺序）
        fp_wt_list = []
        fp_mut_list = []
        valid_indices = []
        
        for idx, row in molecules_df.iterrows():
            mol_id = f"mol_{idx+1}"  # 与 screen_and_collect_fingerprints 生成的 ID 对应
            
            # 查找对应的指纹
            # screen_and_collect_fingerprints 返回的 ID 是从 0 开始的整数字符串
            found_wt = None
            found_mut = None
            
            # 尝试多种可能的 ID 格式
            for possible_id in [str(idx), f"mol_{idx}", f"mol_{idx+1}", mol_id]:
                if possible_id in wt_fp_dict and possible_id in mut_fp_dict:
                    found_wt = wt_fp_dict[possible_id]
                    found_mut = mut_fp_dict[possible_id]
                    break
            
            if found_wt is not None and found_mut is not None:
                fp_wt_list.append(found_wt)
                fp_mut_list.append(found_mut)
                valid_indices.append(idx)
            else:
                print(f"⚠️ 跳过分子 {mol_id}: 缺少指纹")
        
        if len(fp_wt_list) == 0:
            raise RuntimeError("未找到任何有效的指纹配对，请检查分子数据")
        
        fp_wt = np.stack(fp_wt_list, axis=0)
        fp_mut = np.stack(fp_mut_list, axis=0)
        
        print(f"✅ 准备了 {len(fp_wt_list)} 个有效配对")
        print(f"   指纹维度: {fp_wt.shape}")
        
        return fp_wt, fp_mut, valid_indices
    
    def predict_mutations(self, fp_wt, fp_mut):
        """使用孪生塔集成模型进行预测"""
        print("🤖 执行孪生塔集成预测...")
        
        # 调用集成预测函数
        ens_pred, per_seed_pred, ens_prob, per_seed_prob = ensemble_predict(
            device=self.device,
            fp_mut=fp_mut,
            fp_wt=fp_wt,
            weights_root=self.model_dir,
            seed_ids=None,  # 自动发现所有 seed
            seed_dirs=None
        )
        
        print(f"✅ 预测完成")
        print(f"   集成预测: {len(ens_pred)} 个样本")
        if ens_prob is not None:
            print(f"   包含分类概率 (prob_pos)")
        
        return ens_pred, ens_prob, per_seed_pred, per_seed_prob
    
    def create_results_dataframe(self, molecules_df, valid_indices, predictions, probabilities, per_seed_pred, per_seed_prob):
        """创建结果 DataFrame"""
        print("📝 生成结果表格...")
        
        # 创建结果 DataFrame
        results = []
        for i, idx in enumerate(valid_indices):
            row = molecules_df.iloc[idx].copy()
            result_row = {
                'ID': row.get('ID', f"mol_{idx+1}"),
                'Ligand': row['Ligand'],
                'pred_delta': predictions[i],
            }
            
            # 添加 prob_pos（如果有）
            if probabilities is not None:
                result_row['prob_pos'] = probabilities[i]
            
            # 添加每个 seed 的预测（可选）
            if per_seed_pred:
                for seed_id in sorted(per_seed_pred.keys(), key=lambda x: str(x)):
                    result_row[f'pred_seed_{seed_id}'] = per_seed_pred[seed_id][i]
            
            if per_seed_prob and probabilities is not None:
                for seed_id in sorted(per_seed_prob.keys(), key=lambda x: str(x)):
                    result_row[f'prob_seed_{seed_id}'] = per_seed_prob[seed_id][i]
            
            # 添加其他原始列
            for col in molecules_df.columns:
                if col not in result_row and col not in ['ID', 'Ligand', 'smiles']:
                    result_row[col] = row[col]
            
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        
        # 列顺序调整
        base_cols = ['ID', 'Ligand', 'pred_delta']
        if 'prob_pos' in results_df.columns:
            base_cols.append('prob_pos')
        
        # 其他列按字母顺序
        other_cols = [c for c in results_df.columns if c not in base_cols]
        results_df = results_df[base_cols + sorted(other_cols)]
        
        return results_df
    
    def generate_analysis_report(self, results_df, output_dir):
        """生成分析报告"""
        print("📊 生成分析报告...")
        
        report_path = os.path.join(output_dir, 'analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 蛋白突变效应预测分析报告\n\n")
            f.write("## 基本统计\n\n")
            f.write(f"- 总分子数: {len(results_df)}\n")
            f.write(f"- 预测完成: {len(results_df[~results_df['pred_delta'].isna()])}\n\n")
            
            f.write("## ΔΔG 预测分布\n\n")
            f.write(f"- 均值: {results_df['pred_delta'].mean():.4f}\n")
            f.write(f"- 标准差: {results_df['pred_delta'].std():.4f}\n")
            f.write(f"- 最小值: {results_df['pred_delta'].min():.4f}\n")
            f.write(f"- 最大值: {results_df['pred_delta'].max():.4f}\n")
            f.write(f"- 中位数: {results_df['pred_delta'].median():.4f}\n\n")
            
            if 'prob_pos' in results_df.columns:
                f.write("## 突变有利概率 (prob_pos) 分布\n\n")
                f.write(f"- 均值: {results_df['prob_pos'].mean():.4f}\n")
                f.write(f"- 标准差: {results_df['prob_pos'].std():.4f}\n\n")
            
            f.write("## Top 10 有利突变 (ΔΔG 最高)\n\n")
            top10 = results_df.nlargest(10, 'pred_delta')
            f.write("| 排名 | ID | ΔΔG | prob_pos |\n")
            f.write("|------|----|----|----------|\n")
            for rank, (_, row) in enumerate(top10.iterrows(), 1):
                prob_str = f"{row['prob_pos']:.4f}" if 'prob_pos' in row else "N/A"
                f.write(f"| {rank} | {row['ID']} | {row['pred_delta']:.4f} | {prob_str} |\n")
            
            f.write("\n## Top 10 不利突变 (ΔΔG 最低)\n\n")
            bottom10 = results_df.nsmallest(10, 'pred_delta')
            f.write("| 排名 | ID | ΔΔG | prob_pos |\n")
            f.write("|------|----|----|----------|\n")
            for rank, (_, row) in enumerate(bottom10.iterrows(), 1):
                prob_str = f"{row['prob_pos']:.4f}" if 'prob_pos' in row else "N/A"
                f.write(f"| {rank} | {row['ID']} | {row['pred_delta']:.4f} | {prob_str} |\n")
            
            f.write("\n---\n")
            f.write("*报告生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n")
        
        print(f"✅ 分析报告已保存: {report_path}")
        return report_path
    
    def predict(self, wt_sequence, mut_sequence, molecules_csv, output_dir="mutation_predictions", cleanup_temp=True):
        """完整的突变效应预测流程"""
        print("\n" + "="*70)
        print("🧬 蛋白突变效应预测系统 (基于三分支孪生塔)")
        print("="*70 + "\n")
        
        # 1. 验证输入
        print("1️⃣ 验证输入数据...")
        wt_sequence = self.validate_protein_sequence(wt_sequence)
        mut_sequence = self.validate_protein_sequence(mut_sequence)
        
        molecules_df = pd.read_csv(molecules_csv)
        molecules_df = self.validate_molecules(molecules_df)
        
        # 2. 创建工作空间
        print("\n2️⃣ 准备工作环境...")
        temp_dir = self.create_temp_workspace()
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 3. 生成 PSICHIC 指纹
            print("\n3️⃣ 生成 PSICHIC 交互指纹...")
            
            wt_fp_dir = os.path.join(temp_dir, 'wt_fingerprints')
            mut_fp_dir = os.path.join(temp_dir, 'mut_fingerprints')
            
            print("\n   [野生型蛋白]")
            wt_result_df, wt_fp_dict = self.generate_psichic_fingerprints(
                wt_sequence, molecules_df, wt_fp_dir, prefix="WT"
            )
            
            print("\n   [突变型蛋白]")
            mut_result_df, mut_fp_dict = self.generate_psichic_fingerprints(
                mut_sequence, molecules_df, mut_fp_dir, prefix="MUT"
            )
            
            # 4. 准备预测数据
            print("\n4️⃣ 准备预测数据...")
            fp_wt, fp_mut, valid_indices = self.prepare_fingerprints_for_prediction(
                wt_fp_dict, mut_fp_dict, molecules_df
            )
            
            # 5. 执行预测
            print("\n5️⃣ 执行孪生塔集成预测...")
            ens_pred, ens_prob, per_seed_pred, per_seed_prob = self.predict_mutations(fp_wt, fp_mut)
            
            # 6. 生成结果
            print("\n6️⃣ 生成预测结果...")
            results_df = self.create_results_dataframe(
                molecules_df, valid_indices, ens_pred, ens_prob, per_seed_pred, per_seed_prob
            )
            
            # 保存结果
            results_csv = os.path.join(output_dir, 'mutation_predictions.csv')
            results_df.to_csv(results_csv, index=False)
            print(f"✅ 预测结果已保存: {results_csv}")
            
            # 7. 生成分析报告
            print("\n7️⃣ 生成分析报告...")
            report_path = self.generate_analysis_report(results_df, output_dir)
            
            # 8. 清理临时文件
            if cleanup_temp and temp_dir and os.path.exists(temp_dir):
                print("\n8️⃣ 清理临时文件...")
                shutil.rmtree(temp_dir)
                print("✅ 临时文件已清理")
            elif not cleanup_temp:
                print(f"\n💾 临时文件保留在: {temp_dir}")
            
            print("\n" + "="*70)
            print("🎉 预测流程成功完成！")
            print("="*70)
            
            return results_csv, report_path
            
        except Exception as e:
            print(f"\n❌ 预测过程中出错: {e}")
            if cleanup_temp and temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise


def main():
    parser = argparse.ArgumentParser(
        description='蛋白突变效应预测工具 (基于三分支孪生塔)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python predict_protein_mutations.py \\
    --wt_sequence "MKTAYIAKQRQISFVKSHFSRQ..." \\
    --mut_sequence "MKTAYVAKQRQISFVKSHFSRQ..." \\
    --molecules_csv molecules.csv \\
    --model_dir weight \\
    --output_dir mutation_predictions

输入 CSV 格式要求:
  必须包含 'smiles' 或 'Ligand' 列
  可选列: 'ID', 'name' 等
        """
    )
    
    parser.add_argument('--wt_sequence', type=str, required=True,
                       help='野生型蛋白质序列')
    parser.add_argument('--mut_sequence', type=str, required=True,
                       help='突变型蛋白质序列')
    parser.add_argument('--molecules_csv', type=str, required=True,
                       help='分子列表 CSV 文件路径')
    parser.add_argument('--model_dir', type=str, default='weight',
                       help='模型目录（包含 seed_*/model_siamese.pth）')
    parser.add_argument('--output_dir', type=str, default='mutation_predictions',
                       help='输出目录')
    parser.add_argument('--keep_temp', action='store_true',
                       help='保留临时文件（用于调试）')
    parser.add_argument('--psichic_weights_dir', type=str, default=None,
                       help='PSICHIC 预训练权重目录（默认: artifacts/weights）')
    parser.add_argument('--psichic_model_name', type=str, 
                       default='Trained on PDBBind v2020',
                       choices=['Trained on PDBBind v2020', 'Pre-trained on Large-scale Interaction Database'],
                       help='PSICHIC 模型名称')
    
    args = parser.parse_args()
    
    try:
        predictor = ProteinMutationPredictor(
            model_dir=args.model_dir,
            psichic_weights_dir=args.psichic_weights_dir,
            psichic_model_name=args.psichic_model_name
        )
        
        results_csv, report_path = predictor.predict(
            wt_sequence=args.wt_sequence,
            mut_sequence=args.mut_sequence,
            molecules_csv=args.molecules_csv,
            output_dir=args.output_dir,
            cleanup_temp=not args.keep_temp
        )
        
        print(f"\n📄 详细结果: {results_csv}")
        print(f"📋 分析报告: {report_path}")
        
    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

