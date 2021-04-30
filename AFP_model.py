import numpy as np
import torch
import torch.nn as nn
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout


# input: SMILES strings of molecules
# intermediate output: molecule representation

class EE_Predictor(nn.Module):
    """
    Task-oriented Predictor Parameters
    ----------
    pred_input_size : int
    depend on graph_feat_size and the concat manner
    n_tasks : int
    Number of tasks, which is also the output size. Default to 1.
    dropout : float
    Probability for performing the dropout. Default to 0.
    """

    def __init__(self, input_size, n_task=1, dropout=0.):
        super().__init__()
        self.in_size = input_size
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, n_task)
        )

    def forward(self, g_feats, samples):  # , sm_id_dict
        # graph feat -> sample feat
        sample_feats = torch.zeros((len(samples), self.in_size))  # dtype=float
        i = 0
        for sm1_id, sm2_id, met_id, lig_id, sol_id in samples:
            sample_feat = torch.cat((g_feats[sm1_id],
                                     g_feats[sm2_id],
                                     g_feats[met_id],
                                     g_feats[lig_id],
                                     g_feats[sol_id]),
                                    0) if met_id != -1 else torch.cat((g_feats[sm1_id],
                                                                       g_feats[sm2_id],
                                                                       g_feats[lig_id],
                                                                       g_feats[sol_id]),
                                                                      0)
            sample_feats[i] = sample_feat
            i += 1

        return self.layer(sample_feats)


class AFP_EE_Predictor(nn.Module):
    """
    an end-to-end model based on AttentiveFP for regression

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    AttentiveFP Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 pred_input_size=800,  # 4*200 - no metal, 5*200 - metal
                 n_tasks=1,
                 dropout=0.):
        super(AFP_EE_Predictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = EE_Predictor(pred_input_size, n_task=n_tasks)

    def forward(self, g, samples, node_feats, edge_feats, get_node_weight=False):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
            Here, we treat all molecule graphs as a batch.

        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.

        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats, samples), node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats, samples)
