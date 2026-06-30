import os
import pickle
import random
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from rdkit import RDLogger

# 禁用 RDKit 的警告日志
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# 或者使用你代码中原有的 warnings 过滤（对部分 RDKit 警告有效）
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ==========================================
# 1. 宏观特征提取 (Macro/Atom-level Features)
# ==========================================

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_feat_list):
    """
    提取原子级特征，用于宏观流 (Macro Stream) 的 2D 拓扑编码
    """
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_feat_list) + \
              [atom.GetDegree() / 10, atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]

    # 包含手性特征等
    try:
        results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                  [atom.HasProp('_ChiralityPossible')]
    except:
        results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)
    return torch.from_numpy(results)


def edge_features(bond):
    """
    提取键特征
    """
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def get_mol_graph_data(mol, atom_feat_list):
    """
    构建分子的图数据 (PyG Data Format)
    """
    # 1. Node Features
    n_features = [(atom.GetIdx(), atom_features(atom, atom_feat_list)) for atom in mol.GetAtoms()]
    n_features.sort()
    _, n_features = zip(*n_features)
    x = torch.stack(n_features)

    # 2. Edge Index & Edge Features
    if len(mol.GetBonds()) > 0:
        edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol.GetBonds()]
        edge_index = torch.LongTensor([(e[0], e[1]) for e in edges]).T
        edge_attr = torch.FloatTensor([e[2:] for e in edges])

        # 转换为无向图 (双向边)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.LongTensor([[], []])
        edge_attr = torch.FloatTensor([])

    return x, edge_index, edge_attr


# ==========================================
# 2. 微观模体提取 (Micro/Motif-level Features)
# ==========================================

def get_brics_motifs(mol):
    """
    基于 BRICS 算法将分子解构为子结构序列 (Motif Stream)
    修复了 TypeError 和索引越界问题
    """
    if mol is None:
        return [], []

    # 1. 查找 BRICS 键
    # BRICS.FindBRICSBonds 返回的是: [((atom1, atom2), (type1, type2)), ...]
    broken_bonds_info = list(BRICS.FindBRICSBonds(mol))

    if len(broken_bonds_info) == 0:
        # 如果没有 BRICS 键，整个分子作为一个模体
        return [mol], [[a.GetIdx() for a in mol.GetAtoms()]]

    # 2. 【关键修复】将原子对 (u, v) 转换为键的索引 (bond index)
    bond_indices = []
    for bond_info in broken_bonds_info:
        # bond_info[0] 是原子索引对 (u, v)
        u, v = bond_info[0]
        bond = mol.GetBondBetweenAtoms(int(u), int(v))
        if bond:
            bond_indices.append(bond.GetIdx())

    # 3. 执行切割
    # addDummies=True 会在断点处增加虚拟原子（通常标记为 *）
    mol_broken = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)

    # 4. 获取碎片及其对应的原子索引
    # 注意：mol_broken 中的原子索引包含原来的原子 + 新增的虚拟原子
    # 虚拟原子的索引通常在最后，且 >= original_num_atoms
    frags_mols = Chem.GetMolFrags(mol_broken, asMols=True)
    frags_indices_raw = Chem.GetMolFrags(mol_broken, asMols=False)

    # 5. 【关键优化】过滤掉虚拟原子的索引
    # 目的：确保返回的索引能对应回原始分子 (Uni-Mol 特征对应的索引)
    original_num_atoms = mol.GetNumAtoms()
    clean_frags_indices = []

    for indices in frags_indices_raw:
        # 只保留原始分子的原子索引 (index < original_num_atoms)
        valid_indices = [i for i in indices if i < original_num_atoms]
        if valid_indices:
            clean_frags_indices.append(valid_indices)
        else:
            # 理论上不应该出现全是虚拟原子的碎片，但在极端边缘情况防守一下
            clean_frags_indices.append([])

    return list(frags_mols), clean_frags_indices


def get_motif_topology(mol, frags_indices):
    """
    根据原始分子的化学键连接，构建模体间的边索引
    """
    if mol is None or not frags_indices:
        return torch.LongTensor([[], []])

    # 1. 构建原子到模体的映射 {atom_idx: motif_idx}
    atom_to_motif = {}
    for m_idx, indices in enumerate(frags_indices):
        for a_idx in indices:
            atom_to_motif[a_idx] = m_idx

    edges = set()
    num_motifs = len(frags_indices)

    # 2. 遍历原始分子的所有键
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # 检查键的两端是否属于记录在案的模体
        if u in atom_to_motif and v in atom_to_motif:
            m_u = atom_to_motif[u]
            m_v = atom_to_motif[v]

            # 如果两端原子属于不同的模体，说明这两个模体在原分子中相连
            if m_u != m_v:
                # 添加无向边
                edges.add((m_u, m_v))
                edges.add((m_v, m_u))

    # 3. 转换为 Tensor
    if len(edges) > 0:
        row = [e[0] for e in edges]
        col = [e[1] for e in edges]
        edge_index = torch.LongTensor([row, col])
    else:
        # 如果没有边 (例如单个模体，或所有模体均不相连-虽然这在化学上少见)
        edge_index = torch.LongTensor([[], []])

    return edge_index
# ==========================================
# 3. 数据集与加载器 (Dataset & DataLoader)
# ==========================================

class DrugDataset(Dataset):
    def __init__(self, tri_list, drug_id_map, macro_graph_dict, uni_mol_dict, uni_atom_dict, motif_dict):
        """
        Args:
            tri_list: [(h_id, t_id, label), ...]
            drug_id_map: {drug_id: smiles}
            macro_graph_dict: {drug_id: (x, edge_index, edge_attr)} - 宏观图数据
            uni_mol_dict: {drug_id: vector} - Uni-Mol 全局 3D 表征
            uni_atom_dict: {drug_id: matrix} - Uni-Mol 原子级特征
            motif_dict: {drug_id: (motif_mols, motif_atom_indices, motif_edge_index)}
        """
        self.tri_list = tri_list
        self.drug_id_map = drug_id_map
        self.macro_graph_dict = macro_graph_dict
        self.uni_mol_dict = uni_mol_dict
        self.uni_atom_dict = uni_atom_dict
        self.motif_dict = motif_dict

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def _build_macro_data(self, drug_id):
        # 构建宏观流数据对象 (保持不变)
        x, edge_index, edge_attr = self.macro_graph_dict[drug_id]
        target_3d = torch.tensor(self.uni_mol_dict.get(drug_id, np.zeros(1536)), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, target_3d=target_3d)

    def _build_motif_data(self, drug_id):
        """
        [修改版] 模体图构建：
        1. 节点 = 模体 (Motif)
        2. 特征 = 模体内所有原子的 Uni-Mol 特征取平均 (Mean Pooling)
        3. 边 = 模体间的拓扑连接
        """
        # 获取 Uni-Mol 原子特征 [1, N_atoms, Feat_Dim]
        uni_atom_feats = self.uni_atom_dict.get(drug_id)
        if uni_atom_feats is None:
            # 这里的 1536 需要根据你实际的 Uni-Mol 维度调整，通常 base 是 512，large 是 1024 或 1536
            uni_atom_feats = np.zeros((1, 1536))

            # 确保转换为 Tensor 且去掉 batch 维度 -> [N_atoms, Feat_Dim]
        uni_atom_feats = torch.tensor(uni_atom_feats, dtype=torch.float)
        if uni_atom_feats.dim() == 3:
            uni_atom_feats = uni_atom_feats[0]

        # 解包预处理好的数据
        # motif_dict 存储结构: (frags_mols, frags_indices, motif_edge_index)
        motifs, atom_indices_list, topology_edge_index = self.motif_dict[drug_id]

        motif_features_list = []

        # 遍历每一个模体，计算聚合特征
        for atom_indices in atom_indices_list:
            # 筛选有效的原子索引 (防止越界)
            valid_indices = [i for i in atom_indices if i < uni_atom_feats.size(0)]

            if not valid_indices:
                # 异常处理：如果模体没有对应的 Uni-Mol 特征，填充零向量
                motif_features_list.append(torch.zeros(uni_atom_feats.size(1)))
                continue

            # 获取该模体下所有原子的特征 [Num_Motif_Atoms, Feat_Dim]
            current_feats = uni_atom_feats[valid_indices]

            # === 核心修改：求平均 (Mean Pooling) ===
            # 将多原子的特征压缩为一个向量，代表该模体
            motif_mean_feat = torch.mean(current_feats, dim=0)
            motif_features_list.append(motif_mean_feat)

        # 堆叠所有模体特征 -> [Num_Motifs, Feat_Dim]
        if motif_features_list:
            x = torch.stack(motif_features_list)
            motif_edge_index = topology_edge_index
            num_motifs = x.size(0)
        else:
            # 极少数情况：空分子
            x = torch.zeros((0, uni_atom_feats.size(1)))
            motif_edge_index = torch.LongTensor([[], []])
            num_motifs = 0

        # 返回 PyG Data 对象
        # 注意：这里不再需要 atom_to_motif_batch，因为现在的节点直接就是模体本身
        return Data(
            x=x,
            edge_index=motif_edge_index,
            num_motifs=num_motifs
        )

    def collate_fn(self, batch):
        # batch: [(h, t, r), ...]
        h_ids, t_ids, rels = zip(*batch)

        # 1. 宏观图 Batch
        h_macro = Batch.from_data_list([self._build_macro_data(i) for i in h_ids])
        t_macro = Batch.from_data_list([self._build_macro_data(i) for i in t_ids])

        # 2. 模体 Batch
        h_motif = Batch.from_data_list([self._build_motif_data(i) for i in h_ids])
        t_motif = Batch.from_data_list([self._build_motif_data(i) for i in t_ids])

        # 3. 标签
        rels = torch.LongTensor(rels)

        return h_macro, t_macro, h_motif, t_motif, rels

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


# ==========================================
# 4. 主预处理流程 (Main Preprocessing)
# ==========================================

def load_data(args, batch_size, fold_i):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, "dataset", args.dataset)
    # 定义缓存文件路径
    cache_dir = f'{dataset_dir}/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'preprocessed_data_fold{fold_i}.pkl')

    # 1. 尝试从本地加载缓存
    if os.path.exists(cache_file):
        print(f"Loading cached preprocessed data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # 从缓存中恢复所有变量
        drug_map = cache_data['drug_map']
        macro_graph_cache = cache_data['macro_graph_cache']
        motif_cache = cache_data['motif_cache']
        uni_mol_dict = cache_data['uni_mol_dict']
        uni_atom_dict = cache_data['uni_atom_dict']
    else:
        print(f"No cache found. Preprocessing molecules for {args.dataset}...")

        # --- 原有的预处理逻辑开始 ---
        # 1. 加载 SMILES
        df_drugs = pd.read_csv(f'{dataset_dir}/drug_smiles.csv', dtype=str)
        drug_map = {row['drug_id']: row['smiles'] for _, row in df_drugs.iterrows()}
        drug_mols = {d_id: Chem.MolFromSmiles(s) for d_id, s in drug_map.items() if s is not None}

        # 2. 确定原子特征列表
        atom_feat_list = set()
        for mol in drug_mols.values():
            if mol:
                for atom in mol.GetAtoms():
                    atom_feat_list.add(atom.GetSymbol())
        atom_feat_list = sorted(list(atom_feat_list))

        # 3. 加载 Uni-Mol 特征
        unimol_path = os.path.join(BASE_DIR, 'unimol_feature')
        with open(os.path.join(unimol_path, f'{args.dataset}_molecular_features.pkl'), 'rb') as f:
            uni_mol_dict = pickle.load(f)
        with open(os.path.join(unimol_path, f'{args.dataset}_atomic_features.pkl'), 'rb') as f:
            uni_atom_dict = pickle.load(f)

        # 4. 预计算：宏观图特征 & BRICS 模体分解
        macro_graph_cache = {}
        motif_cache = {}

        for d_id, mol in tqdm(drug_mols.items()):
            if mol is None: continue
            macro_graph_cache[d_id] = get_mol_graph_data(mol, atom_feat_list)
            frags_mols, frags_indices = get_brics_motifs(mol)
            motif_edge_index = get_motif_topology(mol, frags_indices)
            motif_cache[d_id] = (frags_mols, frags_indices, motif_edge_index)

        # --- 保存到缓存 ---
        print(f"Saving preprocessed data to {cache_file}...")
        cache_to_save = {
            'drug_map': drug_map,
            'macro_graph_cache': macro_graph_cache,
            'motif_cache': motif_cache,
            'uni_mol_dict': uni_mol_dict,
            'uni_atom_dict': uni_atom_dict
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_to_save, f)
        # --- 原有的预处理逻辑结束 ---

    # 5. 加载分割数据 (这部分不建议缓存，因为 fold 可能变化)
    def load_triplets(file_path):
        df = pd.read_csv(file_path)
        return [(row['Drug1_ID'], row['Drug2_ID'], int(row['Y'])) for _, row in df.iterrows()]

    train_tri = load_triplets(f'{dataset_dir}/train_fold{fold_i}.csv')
    val_tri = load_triplets(f'{dataset_dir}/test_fold{fold_i}.csv')
    test_tri = load_triplets(f'{dataset_dir}/test_fold{fold_i}.csv')

    # 6. 构建 Dataset 和 DataLoader
    train_ds = DrugDataset(train_tri, drug_map, macro_graph_cache, uni_mol_dict, uni_atom_dict, motif_cache)
    val_ds = DrugDataset(val_tri, drug_map, macro_graph_cache, uni_mol_dict, uni_atom_dict, motif_cache)
    test_ds = DrugDataset(test_tri, drug_map, macro_graph_cache, uni_mol_dict, uni_atom_dict, motif_cache)

    train_loader = DrugDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DrugDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DrugDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
# 使用示例
# args = type('Args', (), {'dataset': 'deng'})()
# train, val, test = load_data(args, batch_size=256, fold_i=0)