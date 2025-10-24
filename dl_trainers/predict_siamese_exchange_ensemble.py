#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用三分支孪生塔回归器（带交换一致性）进行推理的脚本。

功能：
- 读取待筛选 CSV（至少包含列：Protein, Ligand），对同一配体的野生型与突变型蛋白分别利用 PSICHIC 生成交互指纹
- 组合为 [mut, wt, diff] 三分支特征，并加载三个权重（不同 seed）与其各自 scaler 做标准化
- 自动从权重 state_dict 中推断网络输入维度与隐藏层配置，必要时对指纹做 TruncatedSVD 以匹配输入维度
- 输出三模型平均后的集成回归预测；若权重内含分类头（多任务），同时输出分类概率（正类 prob）

示例：
  python dl_trainers/predict_siamese_exchange_ensemble.py \
    --input_csv LB0811/LB_stage2/LBstage2.csv \
    --weights_root dl_results_siamese \
    --wt_seq "MAAAAA..." \
    --mut_seq "MBBB..." \
    --output_dir dl_infer_results --use_gpu
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
import pickle

# 确保可以从项目根目录导入模块
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CUR_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# 复用训练时的模型结构（同目录）
from train_siamese_exchange import SiameseRegressor

# 复用 PSICHIC 指纹生成流程（项目根下的 utils / models）
from utils.utils import DataLoader, virtual_screening
from utils.dataset import ProteinMoleculeDataset
from utils import protein_init, ligand_init
from models.net import net as PSICHICNet


def infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, List[int], int, bool]:
    """从保存的 state_dict 中推断 in_dim、hidden_dims、fusion_dim。"""
    # encoder: Linear(in_dim -> h1), Linear(h1 -> h2), ...
    # 依次查找 encoder.net.*.weight 的线性层权重
    hidden_dims: List[int] = []
    in_dim = None
    idx = 0
    while True:
        key = f"encoder.net.{idx}.weight"
        if key not in state_dict:
            break
        weight = state_dict[key]
        out_dim, in_dim_candidate = weight.shape
        if in_dim is None:
            in_dim = int(in_dim_candidate)
        hidden_dims.append(int(out_dim))
        # 跳过 BN/GELU/Dropout 三个层，线性层步进 4
        idx += 4

    if in_dim is None or not hidden_dims:
        raise ValueError("无法从 state_dict 推断网络结构：缺少 encoder 层权重")

    # regressor.0: Linear(hidden[-1]*3 -> fusion_dim)
    reg0_key = "regressor.0.weight"
    if reg0_key not in state_dict:
        raise ValueError("无法从 state_dict 推断 fusion_dim：缺少 regressor.0 权重")
    fusion_dim = int(state_dict[reg0_key].shape[0])
    has_cls = any(k.startswith('cls_head.') for k in state_dict.keys())
    return in_dim, hidden_dims, fusion_dim, has_cls


def load_one_seed_model(seed_dir: str, device: torch.device) -> Tuple[SiameseRegressor, Dict[str, object], bool]:
    """加载单个 seed 的模型与标准化器。
    返回：(model, scalers_dict)
    scalers_dict 包含 'scaler_m','scaler_w','scaler_d'
    """
    model_path = os.path.join(seed_dir, "model_siamese.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    in_dim, hidden_dims, fusion_dim, has_cls = infer_arch_from_state_dict(state_dict)

    model = SiameseRegressor(
        in_dim=in_dim,
        hidden=hidden_dims,
        fusion_dim=fusion_dim,
        dropout=0.2,
        enable_multitask=has_cls,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scalers = {}
    for key in ["scaler_m", "scaler_w", "scaler_d"]:
        pkl_path = os.path.join(seed_dir, f"{key}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"未找到标准化器: {pkl_path}")
        with open(pkl_path, "rb") as f:
            scalers[key] = pickle.load(f)
    return model, scalers, has_cls


def init_psichic(device: torch.device, model_name: str = 'Trained on PDBBind v2020', base_trained_weights_dir: str = os.path.join('artifacts','weights')) -> Tuple[PSICHICNet, Dict]:
    """按 fp.py 的方式加载 PSICHIC 模型与配置。"""
    if model_name == 'Trained on PDBBind v2020':
        trained_model_subpath = 'PDBv2020_PSICHIC'
    elif model_name == 'Pre-trained on Large-scale Interaction Database':
        trained_model_subpath = 'multitask_PSICHIC'
    else:
        raise ValueError(f"未知 PSICHIC 模型: {model_name}")

    trained_model_path = os.path.join(base_trained_weights_dir, trained_model_subpath)
    config_path = os.path.join(trained_model_path, 'config.json')
    degree_path = os.path.join(trained_model_path, 'degree.pt')
    model_pt_path = os.path.join(trained_model_path, 'model.pt')

    if not os.path.isdir(trained_model_path):
        raise FileNotFoundError(f"PSICHIC 目录不存在: {trained_model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"PSICHIC 配置缺失: {config_path}")
    if not os.path.exists(degree_path):
        raise FileNotFoundError(f"PSICHIC 度文件缺失: {degree_path}")
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"PSICHIC 权重缺失: {model_pt_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    degree_dict = torch.load(degree_path, map_location=device)
    if not isinstance(degree_dict, dict) or 'ligand_deg' not in degree_dict or 'protein_deg' not in degree_dict:
        raise RuntimeError("PSICHIC degree.pt 内容非法")
    mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']

    model = PSICHICNet(
        mol_deg, prot_deg,
        mol_in_channels=config['params']['mol_in_channels'], prot_in_channels=config['params']['prot_in_channels'],
        prot_evo_channels=config['params']['prot_evo_channels'],
        hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
        post_layers=config['params']['post_layers'], aggregators=config['params']['aggregators'],
        scalers=config['params']['scalers'], total_layer=config['params']['total_layer'],
        K=config['params']['K'], heads=config['params']['heads'],
        dropout=config['params']['dropout'], dropout_attn_score=config['params']['dropout_attn_score'],
        regression_head=config['tasks']['regression_task'],
        classification_head=config['tasks']['classification_task'],
        multiclassification_head=config['tasks']['mclassification_task'],
        device=device
    ).to(device)
    model.reset_parameters()
    model.load_state_dict(torch.load(model_pt_path, map_location=device))
    model.eval()
    return model, config


def screen_and_collect_fingerprints(ligands: List[str], protein_seq: str, device: torch.device,
                                    result_dir: str, batch_size: int = 16,
                                    save_interpret: bool = True) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """给定配体列表与蛋白序列，使用 PSICHIC 生成指纹，并读取 fingerprint.npy 为内存字典。
    返回：(带 ID 的结果 DataFrame, {ID: fp})
    """
    os.makedirs(result_dir, exist_ok=True)

    # 过滤非法/空 SMILES
    from rdkit import Chem  # 局部导入，避免全局依赖问题
    cleaned_ligands: List[str] = []
    for smi in ligands:
        if smi is None:
            continue
        s = str(smi).strip()
        if s == '' or s.lower() in {'nan', 'none', 'null'}:
            continue
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        if mol is None:
            continue
        cleaned_ligands.append(s)

    if len(cleaned_ligands) == 0:
        raise RuntimeError('提供的配体列表均无效，无法生成指纹')

    # 初始化 PSICHIC 输入数据
    screen_df = pd.DataFrame({
        'Protein': [protein_seq for _ in cleaned_ligands],
        'Ligand': cleaned_ligands,
    })

    # 初始化图字典
    protein_dict = protein_init(screen_df['Protein'].astype(str).unique().tolist())
    ligand_dict = ligand_init(screen_df['Ligand'].astype(str).unique().tolist())

    dataset = ProteinMoleculeDataset(screen_df, ligand_dict, protein_dict, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

    # 加载 PSICHIC 模型
    psichic_root = getattr(screen_and_collect_fingerprints, '_psichic_root', os.path.join('artifacts','weights'))
    psichic_model_name = getattr(screen_and_collect_fingerprints, '_psichic_model', 'Trained on PDBBind v2020')
    psichic_model, _ = init_psichic(device, model_name=psichic_model_name, base_trained_weights_dir=psichic_root)

    # 运行筛选并保存解释（fingerprint.npy）
    result_df = virtual_screening(screen_df.copy(), psichic_model, loader,
                                  result_path=os.path.join(result_dir, "interpretation_result"),
                                  save_interpret=save_interpret, ligand_dict=ligand_dict, device=device)

    # 收集指纹到内存
    fp_dict: Dict[str, np.ndarray] = {}
    interp_dir = os.path.join(result_dir, "interpretation_result")
    for _id in result_df['ID'].tolist():
        fp_path = os.path.join(interp_dir, _id, 'fingerprint.npy')
        if not os.path.exists(fp_path):
            raise FileNotFoundError(f"缺少指纹文件: {fp_path}")
        fp = np.load(fp_path)
        fp = np.asarray(fp, dtype=np.float32)
        fp_dict[_id] = fp
    return result_df, fp_dict


def load_fp_from_existing(result_dir: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """从已有的 PSICHIC 结果目录读取 screening_results 与 fingerprint.npy。
    目录应包含: screening_results.csv 与 interpretation_result/PAIR_*/fingerprint.npy
    返回: (screen_df, {ID: fp})
    """
    csv_path = os.path.join(result_dir, 'screening_results.csv')
    interp_dir = os.path.join(result_dir, 'interpretation_result')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 {csv_path}")
    if not os.path.isdir(interp_dir):
        raise FileNotFoundError(f"未找到 {interp_dir}")
    df = pd.read_csv(csv_path)
    if 'ID' not in df.columns or 'Ligand' not in df.columns:
        raise ValueError(f"{csv_path} 缺少必要列 'ID' 或 'Ligand'")
    fp_dict: Dict[str, np.ndarray] = {}
    for _id in df['ID'].astype(str).tolist():
        fp_path = os.path.join(interp_dir, _id, 'fingerprint.npy')
        if not os.path.exists(fp_path):
            raise FileNotFoundError(f"缺少指纹文件: {fp_path}")
        fp = np.load(fp_path)
        fp_dict[_id] = np.asarray(fp, dtype=np.float32)
    return df, fp_dict


def fit_optional_svd(X: np.ndarray, target_dim: int) -> Tuple[TruncatedSVD, np.ndarray]:
    """如需，将特征 X 通过 SVD 降到 target_dim。返回 (svd, X_transformed)。
    若 X.shape[1] == target_dim，则跳过并返回 (None, X)。
    """
    in_dim = X.shape[1]
    if in_dim == target_dim:
        return None, X
    if in_dim < target_dim:
        raise ValueError(f"特征维度 {in_dim} 小于目标维度 {target_dim}，无法通过SVD升维")
    svd = TruncatedSVD(n_components=target_dim, random_state=0)
    X_t = svd.fit_transform(X)
    return svd, X_t


def predict_with_one_model(model: SiameseRegressor, scalers: Dict[str, object], device: torch.device,
                           fp_mut: np.ndarray, fp_wt: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """用单个 seed 模型进行推理，返回 (回归预测, 分类概率或None)。
    自动匹配输入维度，必要时对 (mut, wt, diff) 分支分别做 SVD。
    """
    assert fp_mut.shape == fp_wt.shape
    base_dim = fp_mut.shape[1]
    diff = fp_mut - fp_wt

    # 推断目标输入维度
    first_linear_weight = model.encoder.net[0].weight  # type: ignore
    target_in_dim = int(first_linear_weight.shape[1])

    # 为三个分支分别拟合/应用 SVD（若需要）
    svd_m, m_t = fit_optional_svd(fp_mut, target_in_dim)
    svd_w, w_t = fit_optional_svd(fp_wt, target_in_dim)
    svd_d, d_t = fit_optional_svd(diff, target_in_dim)

    # 标准化（使用训练时保存的 scaler）
    m_t = scalers['scaler_m'].transform(m_t)
    w_t = scalers['scaler_w'].transform(w_t)
    d_t = scalers['scaler_d'].transform(d_t)

    # 转为张量并前向
    m = torch.from_numpy(m_t).float().to(device)
    w = torch.from_numpy(w_t).float().to(device)
    d = torch.from_numpy(d_t).float().to(device)
    model.eval()
    with torch.no_grad():
        pred, logits = model(m, w, d)
        reg = pred.squeeze(-1).detach().cpu().numpy().reshape(-1)
        prob = None
        if logits is not None:
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    return reg, prob


def ensemble_predict(device: torch.device,
                     fp_mut: np.ndarray, fp_wt: np.ndarray,
                     weights_root: str = None,
                     seed_ids: List[int] = None,
                     seed_dirs: List[str] = None) -> Tuple[np.ndarray, Dict[int, np.ndarray], Optional[np.ndarray], Optional[Dict[int, np.ndarray]]]:
    """加载多个权重目录做推理。
    返回: (回归集成均值, 回归每seed, 分类概率集成均值或None, 概率每seed或None)
    优先使用 seed_dirs（显式传入的目录列表），否则从 weights_root 自动发现 seed_* 目录。
    """
    discovered: List[str] = []
    if seed_dirs:
        for d in seed_dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"权重目录不存在: {d}")
            discovered.append(d)
    else:
        if not weights_root or not os.path.isdir(weights_root):
            raise FileNotFoundError(f"未找到权重根目录，请确认 --weights_root 路径存在: {weights_root}")
        # 自动发现 seed 目录
        candidates = []
        for name in sorted(os.listdir(weights_root)):
            if name.startswith('seed_'):
                sub = os.path.join(weights_root, name)
                if os.path.isdir(sub):
                    candidates.append(sub)
        if not candidates:
            raise FileNotFoundError(f"在 {weights_root} 下未发现 seed_* 目录，请检查训练输出或改用 --seed_dirs 显式传入")
        discovered = sorted(candidates)[:3]  # 仅取前三个

    per_seed_pred: Dict[int, np.ndarray] = {}
    per_seed_prob: Dict[int, np.ndarray] = {}
    any_prob = False
    for sd_dir in discovered:
        # 提取 seed id（如可解析）
        base = os.path.basename(sd_dir.rstrip('/\\'))
        try:
            sd = int(base.split('_', 1)[1]) if base.startswith('seed_') else base
        except Exception:
            sd = base
        model, scalers, has_cls = load_one_seed_model(sd_dir, device)
        reg, prob = predict_with_one_model(model, scalers, device, fp_mut, fp_wt)
        per_seed_pred[sd] = reg
        if prob is not None:
            per_seed_prob[sd] = prob
            any_prob = True
    # 对齐长度并求均值
    preds = np.stack([per_seed_pred[k] for k in sorted(per_seed_pred.keys(), key=lambda x: str(x))], axis=1)
    ens = preds.mean(axis=1)
    if any_prob:
        probs = np.stack([per_seed_prob[k] for k in sorted(per_seed_prob.keys(), key=lambda x: str(x))], axis=1)
        ens_prob = probs.mean(axis=1)
    else:
        ens_prob = None
        per_seed_prob = None  # type: ignore
    return ens, per_seed_pred, ens_prob, per_seed_prob


def parse_args():
    p = argparse.ArgumentParser("Siamese ensemble inference")
    p.add_argument('--input_csv', type=str, required=True, help='待筛选 CSV 路径（需含列 Protein, Ligand）')
    p.add_argument('--weights_root', type=str, default='dl_results_for_LB_mutation', help='训练输出目录（包含 seed_*/model_siamese.pth 与 scaler_*.pkl），默认 dl_results_for_LB_mutation')
    p.add_argument('--wt_seq', type=str, default='MAALSGGGGGGAEPGQALFNGDMEPEAGAGAGAAASSAADPAIPEEVWNIKQMIKLTQEHIEALLDKFGGEHNPPSIYLEAYEEYTSKLDALQQREQQLLESLGNGTDFSVSSSASMDTVTSSSSSSLSVLPSSLSVFQNPTDVARSNPKSPQKPIVRVFLPNKQRTVVPARCGVTVRDSLKKALMMRGLIPECCAVYRIQDGEKKPIGWDTDISWLTGEELHVEVLENVPLTTHNFVRKTFFTLAFCDFCRKLLFQGFRCQTCGYKFHQRCSTEVPLMCVNYDQLDLLFVSKFFEHHPIPQEEASLAETALTSGSSPSAPASDSIGPQILTSPSPSKSIPIPQPFRPADEDHRNQFGQRDRSSSAPNVHINTIEPVNIDDLIRDQGFRGDGGSTTGLSATPPASLPGSLTNVKALQKSPGPQRERKSSSSSEDRNRMKTLGRRDSSDDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAVKMLNVTAPTPQQLQAFKNEVGVLRKTRHVNILLFMGYSTKPQLAIVTQWCEGSSLYHHLHIIETKFEMIKLIDIARQTAQGMDYLHAKSIIHRDLKSNNIFLHEDLTVKIGDFGLATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQILASIELLARSLPKIHRSASEPSLNRAGFQTEDFSLYACASPKTPIQAGGYGAFPVH', help='野生型蛋白序列（默认已内置）')
    p.add_argument('--mut_seq', type=str, default='MAALSGGGGGGAEPGQALFNGDMEPEAGAGAGAAASSAADPAIPEEVWNIKQMIKLTQEHIEALLDKFGGEHNPPSIYLEAYEEYTSKLDALQQREQQLLESLGNGTDFSVSSSASMDTVTSSSSSSLSVLPSSLSVFQNPTDVARSNPKSPQKPIVRVFLPNKQRTVVPARCGVTVRDSLKKALMMRGLIPECCAVYRIQDGEKKPIGWDTDISWLTGEELHVEVLENVPLTTHNFVRKTFFTLAFCDFCRKLLFQGFRCQTCGYKFHQRCSTEVPLMCVNYDQLDLLFVSKFFEHHPIPQEEASLAETALTSGSSPSAPASDSIGPQILTSPSPSKSIPIPQPFRPADEDHRNQFGQRDRSSSAPNVHINTIEPVNIDDLIRDQGFRGDGGSTTGLSATPPASLPGSLTNVKALQKSPGPQRERKSSSSSEDRNRMKTLGRRDSSDDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDVAVKMLNVTAPTPQQLQAFKNEVGVLRKTRHVNILLFMGYSTKPQLAIVTQWCEGSSLYHHLHIIETKFEMIKLIDIARQTAQGMDYLHAKSIIHRDLKSNNIFLHEDLTVKIGDFGLATEKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQILASIELLARSLPKIHRSASEPSLNRAGFQTEDFSLYACASPKTPIQAGGYGAFPVH', help='突变型蛋白序列（默认已内置）')
    p.add_argument('--output_dir', type=str, default='dl_siamese_infer', help='推理输出目录')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--use_gpu', action='store_true')
    p.add_argument('--psichic_weights_dir', type=str, default=os.path.join('artifacts','weights'), help='PSICHIC 预训练权重根目录（包含 PDBv2020_PSICHIC 或 multitask_PSICHIC）')
    p.add_argument('--psichic_model_name', type=str, default='Trained on PDBBind v2020', choices=['Trained on PDBBind v2020', 'Pre-trained on Large-scale Interaction Database'])
    p.add_argument('--wt_fp_dir', type=str, default=None, help='可选：已有 WT 指纹结果目录（包含 screening_results.csv 与 interpretation_result）')
    p.add_argument('--mut_fp_dir', type=str, default=None, help='可选：已有 MUT 指纹结果目录（包含 screening_results.csv 与 interpretation_result）')
    p.add_argument('--seed_dirs', type=str, nargs='+', default=None, help='显式传入的权重目录列表（每个目录含 model_siamese.pth 与 scaler_*.pkl）')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # 读取输入 CSV 并提取唯一配体
    df_in = pd.read_csv(args.input_csv)
    if 'Ligand' not in df_in.columns:
        raise ValueError("输入 CSV 缺少 'Ligand' 列")
    ligands = df_in['Ligand'].astype(str).unique().tolist()

    # 分别对 WT 与 MUT 生成指纹
    wt_dir = os.path.join(run_dir, 'psichic_wt')
    mut_dir = os.path.join(run_dir, 'psichic_mut')
    # 配置 PSICHIC 模型根目录和名称（通过函数属性传入）
    screen_and_collect_fingerprints._psichic_root = args.psichic_weights_dir
    screen_and_collect_fingerprints._psichic_model = args.psichic_model_name

    if args.wt_fp_dir and args.mut_fp_dir and os.path.isdir(args.wt_fp_dir) and os.path.isdir(args.mut_fp_dir):
        print('(1) 读取已有 WT/MUT 指纹目录...')
        df_wt, fp_wt_dict = load_fp_from_existing(args.wt_fp_dir)
        df_mut, fp_mut_dict = load_fp_from_existing(args.mut_fp_dir)
    else:
        print('(1) 生成 WT 指纹...')
        df_wt, fp_wt_dict = screen_and_collect_fingerprints(ligands, args.wt_seq, device, wt_dir, batch_size=args.batch_size)
        print('(2) 生成 MUT 指纹...')
        df_mut, fp_mut_dict = screen_and_collect_fingerprints(ligands, args.mut_seq, device, mut_dir, batch_size=args.batch_size)

    # 按 Ligand 对齐顺序，组装特征矩阵
    id_wt_by_ligand = dict(zip(df_wt['Ligand'].astype(str), df_wt['ID'].astype(str)))
    id_mut_by_ligand = dict(zip(df_mut['Ligand'].astype(str), df_mut['ID'].astype(str)))

    fp_list_wt: List[np.ndarray] = []
    fp_list_mut: List[np.ndarray] = []
    kept_ligands: List[str] = []
    for smi in ligands:
        if smi not in id_wt_by_ligand or smi not in id_mut_by_ligand:
            continue
        wid = id_wt_by_ligand[smi]
        mid = id_mut_by_ligand[smi]
        if wid not in fp_wt_dict or mid not in fp_mut_dict:
            continue
        fp_list_wt.append(fp_wt_dict[wid])
        fp_list_mut.append(fp_mut_dict[mid])
        kept_ligands.append(smi)

    if not fp_list_wt or not fp_list_mut:
        raise RuntimeError('未能收集到有效的指纹特征，请检查输入与 PSICHIC 权重是否就绪')

    fp_wt = np.stack(fp_list_wt, axis=0).astype(np.float32)
    fp_mut = np.stack(fp_list_mut, axis=0).astype(np.float32)

    # 集成推理
    print('(3) 加载权重并进行集成推理...')
    ens_pred, per_seed, ens_prob, per_seed_prob = ensemble_predict(device, fp_mut, fp_wt, weights_root=args.weights_root, seed_dirs=args.seed_dirs)

    # 保存结果（去重配体层面）
    out_df = pd.DataFrame({
        'Ligand': kept_ligands,
        'WT_Protein': [args.wt_seq for _ in kept_ligands],
        'MUT_Protein': [args.mut_seq for _ in kept_ligands],
        'pred_delta': ens_pred,
    })
    # 合并原CSV中的 Name（按 SMILES/Ligand 映射）
    if 'Name' in df_in.columns:
        name_map = df_in[['Ligand', 'Name']].copy()
        name_map['Ligand'] = name_map['Ligand'].astype(str)
        name_map = name_map.dropna(subset=['Ligand']).drop_duplicates(subset=['Ligand'])
        out_df = out_df.merge(name_map, on='Ligand', how='left')

    # 附加每个 seed 的预测列
    for sd, arr in per_seed.items():
        out_df[f'pred_seed_{sd}'] = arr

    # 分类概率（如可用）
    if ens_prob is not None:
        out_df['prob_pos'] = ens_prob
        for sd, arr in (per_seed_prob or {}).items():
            out_df[f'prob_seed_{sd}'] = arr

    out_csv_unique = os.path.join(run_dir, 'siamese_ensemble_predictions.csv')
    out_df.to_csv(out_csv_unique, index=False)
    print(f'(4) 完成，结果已保存: {out_csv_unique}')


if __name__ == '__main__':
    main()


