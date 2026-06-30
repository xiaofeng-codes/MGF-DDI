import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GlobalAttention, LayerNorm
from GraphTransformerLayer import GraphSelfAttention, GraphFFN, GraphLayerNorm
from fusion import TriViewFusion
from FuzzyLayer import FuzzyLayer

class GraphTransformerLayer(nn.Module):
    """
    单层 Graph Transformer 模块 (Wrapper)
    """

    def __init__(self, d_hidden, d_edge_hid, d_target_3d, heads=4, dropout=0.5):
        super().__init__()

        # d_edge_hid (64) 用于边特征的维度
        # d_hidden (256) 用于节点特征的维度
        self.d_edge = d_edge_hid
        self.hid_edge = d_edge_hid
        self.target_3d = d_target_3d

        # 1. Self-Attention
        self.attention = GraphSelfAttention(
            d_hidden=d_hidden,
            d_edge=self.d_edge,  # 传递边维度标志/值
            hid_edge=self.hid_edge,  # 传递具体的边隐藏层维度
            d_target_3d=self.target_3d,
            n_head=heads,
            dropout=dropout,
            residual=True
        )

        # 2. Norm 1
        self.norm1 = GraphLayerNorm(d_hidden, self.d_edge, self.hid_edge)

        # 3. FFN
        self.ffn = GraphFFN(
            d_in=d_hidden,
            d_hidden=d_hidden,
            d_out=d_hidden,
            d_edge=self.d_edge,
            hid_edge=self.hid_edge,
            dropout=dropout,
            residual=True
        )

        # 4. Norm 2
        self.norm2 = GraphLayerNorm(d_hidden, self.d_edge, self.hid_edge)

    def forward(self, x, edge_attr, edge_index, target_3d, batch):
        # 1. Attention + Residual
        x, edge_attr = self.attention(x, edge_attr, batch, edge_index, target_3d)

        # 2. Norm 1
        x, edge_attr = self.norm1(x, edge_attr, batch, edge_index)

        # 3. FFN + Residual
        x, edge_attr = self.ffn(x, edge_attr, batch, edge_index)

        # 4. Norm 2
        x, edge_attr = self.norm2(x, edge_attr, batch, edge_index)



        return x, edge_attr


class GraphTransformerEncoder(nn.Module):
    """
    图 Transformer 编码器堆叠
    """

    def __init__(self, d_in, d_hidden, d_edge_in, d_edge_hid, d_target_3d, num_layers=3, heads=4, dropout=0.5):
        super().__init__()

        self.init_norm = LayerNorm(d_in)
        # 节点特征投影 d_in -> 256
        self.input_proj = nn.Linear(d_in, d_hidden)

        self.edge_norm = LayerNorm(d_edge_in)
        # 边特征投影 d_edge_in (6) -> d_edge_hid (64)
        self.edge_proj = nn.Linear(d_edge_in, d_edge_hid)
        # self.edge_act = nn.SiLU()

        # self.target_norm = LayerNorm(d_target_3d)

        self.layers = nn.ModuleList([
            # 每一层处理 d_hidden (256) 的节点和 d_edge_hid (64) 的边
            GraphTransformerLayer(d_hidden, d_edge_hid, d_target_3d, heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr, target_3d, batch):
        # 1. 投影节点特征
        x = self.init_norm(x, batch)
        x = self.input_proj(x)


        # 2. 投影边特征
        edge_id = batch[edge_index[0]]
        edge_attr = self.edge_norm(edge_attr, edge_id)
        edge_attr = self.edge_proj(edge_attr)
        # edge_attr = self.edge_act(edge_attr)

        # target_3d = self.target_norm(target_3d)

        # 3. 堆叠层处理
        for layer in self.layers:
            x, edge_attr = layer(x, edge_attr, edge_index, target_3d, batch)


        return x


class MacroEncoder(nn.Module):
    """
    宏观流编码器
    """

    def __init__(self, d_atom, d_hidden, d_edge_in, d_edge_hid, d_unimol, gt_layers=1):
        super().__init__()

        self.graph_transformer = GraphTransformerEncoder(
            d_in=d_atom,
            d_hidden=d_hidden,  # 256
            d_edge_in=d_edge_in,  # 6
            d_edge_hid=d_edge_hid,  # 64
            d_target_3d=d_unimol,
            num_layers=gt_layers,
            heads=4
        )

        self.distill_proj = nn.Linear(d_hidden, d_unimol)

    def forward(self, data):
        x, edge_index, edge_attr, target_3d, batch = data.x, data.edge_index, data.edge_attr, data.target_3d, data.batch

        # 传入所有必要的图信息
        node_feats = self.graph_transformer(x, edge_index, edge_attr, target_3d, batch)

        # Readout
        macro_rep = global_mean_pool(node_feats, batch)

        return macro_rep, self.distill_proj(macro_rep)

class FuzzyGCN(nn.Module):
    def __init__(self, d_hidden, d_out, mem_num=5):
        super().__init__()
        self.first_norm = LayerNorm(d_hidden)
        self.first_linear = nn.Linear(d_hidden, d_hidden)
        self.output_norm = LayerNorm(d_hidden)

        self.FuzzyLayers = nn.ModuleList([
            FuzzyLayer(d_hidden, d_hidden, 20)
            for _ in range(mem_num)
        ])

        self.GCNConv = GCNConv(d_hidden, d_hidden)

        self.dropout = nn.Dropout(0.5)

    def forward(self, drug, edge_index, batch, edge_attr=None):

        first_out = drug


        gcn_out = self.GCNConv(first_out, edge_index)


        midnorm_out = self.first_norm(gcn_out, batch)

        layer_outputs = [layer(midnorm_out) for layer in self.FuzzyLayers]

        fuzzy_out = torch.stack(layer_outputs, dim=-1)

        outnorm_out = fuzzy_out.sum(dim=-1)

        final_out = outnorm_out * gcn_out

        if torch.isnan(final_out).any() or torch.isinf(final_out).any():
            print(final_out)

        return final_out


class MicroEncoder(nn.Module):
    """
    微观模体流 (Updated to use FuzzyGCN)
    """

    def __init__(self, d_unimol, d_hidden, num_layers=1, mem_num=5, dropout=0.5):
        super().__init__()
        self.init_norm = LayerNorm(d_unimol)
        self.input_proj = nn.Linear(d_unimol, d_hidden)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # 修改处：使用 FuzzyGCN
            self.convs.append(FuzzyGCN(d_hidden, d_hidden))

        self.FGCN = nn.ModuleList([
            FuzzyGCN(d_hidden, d_hidden, mem_num=mem_num) for _ in range(num_layers)  # 3层FuzzyGCN
        ])

        self.dropout = dropout
        self.att_pool = GlobalAttention(gate_nn=nn.Linear(d_hidden, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. 初始化归一化
        x = self.init_norm(x, batch)

        x = self.input_proj(x)

        for layer in self.FGCN:
            x = layer(x, edge_index, batch)

        node_feats = x
        micro_rep = self.att_pool(node_feats, batch)

        return micro_rep, node_feats

class SubstructureInteraction(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.proj = nn.Linear(d_hidden, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, h_nodes, t_nodes, h_batch, t_batch):
        h_pool = global_mean_pool(h_nodes, h_batch)
        t_pool = global_mean_pool(t_nodes, t_batch)
        interaction = h_pool * t_pool
        return self.norm(self.proj(interaction))


class DynamicFusion(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_hidden * 3, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 3)
        )

    def forward(self, v_macro, v_micro, v_inter):
        stack = torch.stack([v_macro, v_micro, v_inter], dim=1)
        concat = torch.cat([v_macro, v_micro, v_inter], dim=-1)

        weights = F.softmax(self.scorer(concat), dim=-1).unsqueeze(-1)
        fused = (stack * weights).sum(dim=1)
        return fused, weights


class MGF_DDI(nn.Module):
    def __init__(self, d_atom=37, d_edge_in=6, d_hidden=256, d_edge_hid=64, d_unimol=1536, rel_total=86,
                 micro_layers=3,   # 现有参数 (GCN层数)
                 gt_layers=3,
                 mem_num=5, dropout=0.5):
        super().__init__()

        # 1. 宏观流
        self.macro_encoder = MacroEncoder(
            d_atom=d_atom,
            d_hidden=d_hidden,
            d_edge_in=d_edge_in,
            d_edge_hid=d_edge_hid,
            d_unimol=d_unimol,
            gt_layers=gt_layers
        )

        # 2. 微观流
        self.micro_encoder = MicroEncoder(
            d_unimol=d_unimol,
            d_hidden=d_hidden,
            num_layers=micro_layers, # 使用传入的 micro_layers
            mem_num=mem_num,
            dropout=dropout
        )

        # 3. 交互模块
        self.interaction = SubstructureInteraction(d_hidden)

        # 4. 融合模块 (TriViewFusion)
        self.fusion = TriViewFusion(
            input_dim=d_hidden * 2,
            hidden_dim=d_hidden * 2,
            num_heads=4,
            dropout=dropout
        )

        # 5. 二分类器 (Binary Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            # [修改] 最终输出维度为 1 (用于 BCEWithLogitsLoss)
            nn.Linear(d_hidden, 1)
        )

        # 投影头 (用于对比学习/辅助损失)
        self.proj_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden)
        )

    def forward(self, h_macro, t_macro, h_motif, t_motif):
        # A. Macro Stream
        h_macro_rep, h_distill = self.macro_encoder(h_macro)
        t_macro_rep, t_distill = self.macro_encoder(t_macro)

        # B. Micro Stream
        h_micro_rep, h_micro_nodes = self.micro_encoder(h_motif)
        t_micro_rep, t_micro_nodes = self.micro_encoder(t_motif)

        # C. Interaction (子结构交互)
        inter_h = self.interaction(h_micro_nodes, t_micro_nodes, h_motif.batch, t_motif.batch)
        # 构造 v_inter: (Batch, Hidden*2)
        v_inter = torch.cat([inter_h, inter_h], dim=-1)

        # D. View Construction
        # 拼接头尾药物特征
        v_macro = torch.cat([h_macro_rep, t_macro_rep], dim=-1)
        v_micro = torch.cat([h_micro_rep, t_micro_rep], dim=-1)

        # E. Fusion & Predict
        final_rep, weights = self.fusion(v_macro, v_micro, v_inter)

        # [修改] Logits 输出 shape 为 (Batch, 1)
        logits = self.classifier(final_rep)

        # F. Auxiliary Loss (对比损失)
        # 计算宏观和微观表征的一致性
        z_macro = self.proj_head(h_macro_rep)
        z_micro = self.proj_head(h_micro_rep)
        loss_cl = -F.cosine_similarity(z_macro, z_micro).mean()

        return logits, loss_cl