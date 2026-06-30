import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std
from torch_geometric.nn import LayerNorm


# 移除了 RadialBasis 和 stable_norm，因为不再处理几何距离

class GraphSelfAttention(nn.Module):
    '''
    Standard Graph Transformer Layer (Non-Equivariant)
    Removed Z (coordinates) and RadialBasis inputs.
    '''

    def __init__(self, d_hidden, d_edge, hid_edge, d_target_3d, n_head, act_fn=nn.SiLU(), residual=True, dropout=0.0):
        super(GraphSelfAttention, self).__init__()

        self.residual = residual
        self.d_head = d_hidden

        self.n_head = n_head
        self.d_edge = d_edge
        self.hid_edge = 0

        self.head_dim = d_hidden // self.n_head
        self.target_3d = d_target_3d

        # 此处可以加入高维目标特征拼接
        self.knowledge_linear = nn.Linear(d_target_3d, d_hidden)
        self.q_linear = nn.Linear(d_hidden * 2, d_hidden)
        self.k_linear = nn.Linear(d_hidden * 2, d_hidden)
        self.v_linear = nn.Linear(d_hidden * 2, d_hidden)

        if self.d_edge != 0:
            self.hid_edge = hid_edge

        # Attention MLP 输入维度：(H_q + H_k + edge_attr) // n_head
        # 移除了 n_rbf (径向基特征)
        self.att_mlp = nn.Sequential(
            nn.Linear((d_hidden * 2 + self.hid_edge) // self.n_head, d_hidden * 4),
            act_fn,
            nn.Linear(d_hidden * 4, self.n_head),
        )

        # 移除了 ed_l 和 ind_l (用于处理距离嵌入的线性层)

    def attention(self, H, edge_attr, edges, add_knowledge):
        unit_row, unit_col = edges[0], edges[1]

        H_q = torch.concat([H[unit_col], add_knowledge[unit_col]], dim=-1)
        H_k = torch.concat([H[unit_row], add_knowledge[unit_row]], dim=-1)

        H_q = self.q_linear(H_q).contiguous().view(H_q.shape[0], self.n_head, -1)
        H_k = self.k_linear(H_k).contiguous().view(H_k.shape[0], self.n_head, -1)

        # 移除了 dZ, D (距离计算)

        if self.d_edge != 0:
            edge_attr_reshaped = edge_attr.contiguous().view(edge_attr.shape[0], self.n_head, -1)
            # 仅拼接 Query, Key 和 Edge Features
            R_repr = torch.concat([H_q, H_k, edge_attr_reshaped], dim=-1)
        else:
            R_repr = torch.concat([H_q, H_k], dim=-1)

        R_repr = self.att_mlp(R_repr)

        alpha = scatter_softmax(R_repr, unit_col, dim=0)
        return alpha

    def update(self, H_v, H, alpha, edge_attr, edges):
        # 原 invariant_update 的简化版
        unit_row, unit_col = edges

        # 移除了 ind_l(D) 的调制，直接使用 Value
        H_agg = alpha @ H_v

        # 更新边特征 (可选)
        if self.d_edge != 0:
            edge_agg = (alpha @ (edge_attr.contiguous().view(edge_attr.shape[0], self.n_head, -1)))
            edge_H_agg = (edge_agg).contiguous().view(edge_attr.shape[0], -1)
            edge_attr = edge_attr + edge_H_agg if self.residual else edge_H_agg

        # 聚合邻居信息到目标节点
        H_agg = scatter_sum(H_agg, unit_row, dim=0)
        H_agg = H_agg.contiguous().view(H_agg.shape[0], -1)

        H = H + H_agg if self.residual else H_agg

        return H, edge_attr

    def forward(self, H, edge_attr, block_id, edges, target_3d):
        # 移除了输入 Z
        add_knowledge = self.knowledge_linear(target_3d)

        add_knowledge = add_knowledge[block_id]

        alpha = self.attention(H, edge_attr, edges, add_knowledge)

        H_v = self.v_linear(torch.concat([H[edges[1]], add_knowledge[edges[1]]], dim=-1))
        H_v = H_v.contiguous().view(H_v.shape[0], self.n_head, -1)

        H, edge_attr = self.update(H_v, H, alpha, edge_attr, edges)

        # 移除了 equivariant_update

        return H, edge_attr


class GraphFFN(nn.Module):
    '''
    Standard Graph FFN
    Removed geometric gating and coordinate updates.
    '''

    def __init__(self, d_in, d_hidden, d_out, d_edge, hid_edge, act_fn=nn.SiLU(),
                 residual=True, dropout=0.1):
        super().__init__()
        self.residual = residual
        self.d_edge = d_edge
        self.hid_edge = 0

        # 移除了 n_rbf 输入
        input_dim = d_in * 2 + self.hid_edge  # H + H_global (+ edge)

        if self.d_edge:
            self.hid_edge = hid_edge
            input_dim = d_in * 2 + self.hid_edge
            self.mlp_edge = nn.Sequential(
                nn.Linear(input_dim, d_hidden * 4),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(d_hidden * 4, self.hid_edge),
            )

        self.mlp_h = nn.Sequential(
            nn.Linear(input_dim, d_hidden * 4),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 4, d_out),
        )

        # 移除了 mlp_z (坐标更新 MLP) 和 rbf

    def forward(self, H, edge_attr, block_id, edge_id):
        # 移除了 Z 输入和 _radial 计算

        # 计算图/Block级别的全局特征作为 Context
        H_c = scatter_mean(H, block_id, dim=0)[block_id]

        if self.d_edge != 0:
            # inputs: [H, H_global, edge_global]
            inputs = torch.cat([H, H_c, scatter_mean(edge_attr, edge_id[0], dim=0)], dim=-1)
            edge_update = self.mlp_edge(inputs)[edge_id[0]] * edge_attr
            edge_attr = edge_attr + edge_update if self.residual else edge_update
        else:
            inputs = torch.cat([H, H_c], dim=-1)

        H_update = self.mlp_h(inputs)

        H = H + H_update if self.residual else H_update

        # 移除了 Z_update

        return H, edge_attr


class GraphLayerNorm(nn.Module):
    '''
    Standard LayerNorm for Graphs
    Removed Coordinate (Z) rescaling.
    '''

    def __init__(self, d_hidden, d_edge, hid_edge, act_fn=nn.SiLU()):
        super().__init__()

        self.d_edge = d_edge
        if self.d_edge:
            self.hid_edge = hid_edge
            self.layernorm_edge = LayerNorm(self.hid_edge)

        # 移除了 fuse_scale_ffn (用于融合距离特征)
        self.layernorm_H = LayerNorm(d_hidden)

        # 移除了 sigma 和 rbf

    def forward(self, H, edge_attr, block_id, edge_id):
        # 移除了 Z 相关的处理逻辑 (去均值、算方差、重缩放)

        H = self.layernorm_H(H, block_id)

        if self.d_edge != 0:
            edge_attr = self.layernorm_edge(edge_attr, block_id[edge_id[0]])

        return H, edge_attr