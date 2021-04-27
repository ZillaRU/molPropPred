# # % matplotlib
# # inline
# import os
# from rdkit import Chem
# from rdkit import RDPaths
#
# import dgl
# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# # from dgl import model_zoo
# from dgllife.model import model_zoo
# # from dgllife.model.gnn import attentivefp
# from dgllife.utils import mol_to_complete_graph, mol_to_bigraph
#
# from dgllife.utils import atom_type_one_hot
# from dgllife.utils import atom_degree_one_hot
# from dgllife.utils import atom_formal_charge
# from dgllife.utils import atom_num_radical_electrons
# from dgllife.utils import atom_hybridization_one_hot
# from dgllife.utils import atom_total_num_H_one_hot
# from dgllife.utils import CanonicalAtomFeaturizer
# from dgllife.utils import CanonicalBondFeaturizer
# from dgllife.utils import ConcatFeaturizer
# from dgllife.utils import BaseAtomFeaturizer
# from dgllife.utils import BaseBondFeaturizer
#
# from dgllife.utils import one_hot_encoding
# from dgl.data.utils import split_dataset
#
# from functools import partial
# from sklearn.metrics import roc_auc_score

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl import model_zoo
from dgl.data.chem import BaseAtomFeaturizer
from dgl.data.chem import BaseBondFeaturizer
from dgl.data.chem import ConcatFeaturizer
from dgl.data.chem import one_hot_encoding
from dgl.data.chem.utils import atom_degree_one_hot
from dgl.data.chem.utils import atom_formal_charge
from dgl.data.chem.utils import atom_hybridization_one_hot
from dgl.data.chem.utils import atom_num_radical_electrons
from dgl.data.chem.utils import atom_total_num_H_one_hot
from dgl.data.chem.utils import atom_type_one_hot
from dgl.data.chem.utils import mol_to_bigraph
from rdkit import Chem
from torch.utils.data import DataLoader


# from dgl.data.chem.utils import mol_to_complete_graph, mol_to_bigraph


def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


atom_featurizer = BaseAtomFeaturizer(
    {'hv': ConcatFeaturizer([
        partial(atom_type_one_hot, allowable_set=[
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=list(range(6))),
        atom_formal_charge, atom_num_radical_electrons,
        partial(atom_hybridization_one_hot, encode_unknown=True),
        lambda atom: [0],  # A placeholder for aromatic information,
        atom_total_num_H_one_hot, chirality
    ],
    )})
bond_featurizer = BaseBondFeaturizer({
    'he': lambda bond: [0 for _ in range(10)]
})

train_mols = Chem.SDMolSupplier('solubility.train.sdf')
train_smi = [Chem.MolToSmiles(m) for m in train_mols]
train_sol = torch.tensor([float(mol.GetProp('SOL')) for mol in train_mols]).reshape(-1, 1)

test_mols = Chem.SDMolSupplier('solubility.test.sdf')
test_smi = [Chem.MolToSmiles(m) for m in test_mols]
test_sol = torch.tensor([float(mol.GetProp('SOL')) for mol in test_mols]).reshape(-1, 1)

train_graph = [mol_to_bigraph(mol,
                              node_featurizer=atom_featurizer,
                              edge_featurizer=bond_featurizer) for mol in train_mols]

test_graph = [mol_to_bigraph(mol,
                             node_featurizer=atom_featurizer,
                             edge_featurizer=bond_featurizer) for mol in test_mols]


def run_a_train_epoch(n_epochs, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    total_loss = 0
    losses = []

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if torch.cuda.is_available():
            bg.to(torch.device('cuda:0'))
            labels = labels.to('cuda:0')
            masks = masks.to('cuda:0')

        prediction = model(bg, bg.ndata['hv'], bg.edata['he'])
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        # loss = loss_criterion(prediction, labels)
        # print(loss.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item())

    # total_score = np.mean(train_meter.compute_metric('rmse'))
    total_score = np.mean(losses)
    print('epoch {:d}/{:d}, training {:.4f}'.format(epoch + 1, n_epochs, total_score))
    return total_score


model = model_zoo.chem.AttentiveFP(node_feat_size=39,
                              # model = attentivefp.AttentiveFPGNN(node_feat_size=39,
                              edge_feat_size=10,
                              num_layers=2,
                              num_timesteps=2,
                              graph_feat_size=200,
                              output_size=1,
                              dropout=0.2)

train_loader = DataLoader(dataset=list(zip(train_smi, train_graph, train_sol)), batch_size=128,
                          collate_fn=collate_molgraphs)
test_loader = DataLoader(dataset=list(zip(test_smi, test_graph, test_sol)), batch_size=128,
                         collate_fn=collate_molgraphs)

loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (-2.5), weight_decay=10 ** (-5.0), )
n_epochs = 120
epochs = []
scores = []
for e in range(n_epochs):
    score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
    epochs.append(e)
    scores.append(score)
model.eval()
torch.save(model, 'model.pkl')
plt.plot(range(n_epochs), scores,)
plt.title(f'mean loss of training(epoch:{n_epochs})')
plt.savefig(f'mean_loss_{n_epochs}.jpg')

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from IPython.display import display
import matplotlib
import matplotlib.cm as cm
import torch
from rdkit import Chem


def drawmol(idx, dataset, timestep):
    smiles, graph, _ = dataset[idx]
    print(smiles)
    bg = dgl.batch([graph])
    atom_feats, bond_feats = bg.ndata['hv'], bg.edata['he']
    if torch.cuda.is_available():
        print('use cuda')
        bg.to(torch.device('cuda:0'))
        atom_feats = atom_feats.to('cuda:0')
        bond_feats = bond_feats.to('cuda:0')

    _, atom_weights = model(bg, atom_feats, bond_feats, get_node_weight=True)
    assert timestep < len(atom_weights), 'Unexpected id for the readout round'
    atom_weights = atom_weights[timestep]
    min_value = torch.min(atom_weights)
    max_value = torch.max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = cm.get_cmap('bwr')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i].data.item()) for i in range(bg.number_of_nodes())}

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    op = drawer.drawOptions()

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(bg.number_of_nodes()),
                        highlightBonds=[],
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    if torch.cuda.is_available():
        atom_weights = atom_weights.to('cpu')
    return (Chem.MolFromSmiles(smiles), atom_weights.data.numpy(), svg)


target = test_loader.dataset
for i in range(len(target)):
    mol, aw, svg = drawmol(i, target, 0)
    display(SVG(svg))
