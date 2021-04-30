# prepare packages

# prepare data
import csv
from functools import partial
from itertools import islice

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dgllife.utils import smiles_to_bigraph, BaseAtomFeaturizer, ConcatFeaturizer, atom_num_radical_electrons, \
    atom_formal_charge, atom_degree_one_hot, \
    atom_type_one_hot, atom_hybridization_one_hot, atom_total_num_H_one_hot, AttentiveFPBondFeaturizer

from AFP_model import AFP_EE_Predictor
from data_utils import chirality

train_file = 'data/train_no_metal.csv'
test_file = 'data/test_no_metal.csv'
mol_id_dict_filename = 'data/no_metal_sm_to_id.npy'
id_mol_dict_filename = 'data/no_metal_id_to_sm.npy'


# load train & test files
# format (SM1, SM2, metal, ligand, solvent, ee)

def build_mol_dict(fname):
    global sm_to_id_dict, id_to_sm_dict, sm_cnt, sm1_cnt, sm2_cnt, met_cnt, lig_cnt, sol_cnt
    id_id_ee_list = list()
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for row in islice(reader, 1, None):  # if the csv has a header , skip the 1st row
            sm1, sm2, met, lig, sol, ee = row
            if sm1 not in sm_to_id_dict:
                sm_to_id_dict[sm1] = sm_cnt
                id_to_sm_dict[sm_cnt] = sm1
                sm_cnt += 1
                sm1_cnt += 1
            if sm2 not in sm_to_id_dict:
                sm_to_id_dict[sm2] = sm_cnt
                id_to_sm_dict[sm_cnt] = sm2
                sm_cnt += 1
                sm2_cnt += 1
            if met != '' and met not in sm_to_id_dict:
                sm_to_id_dict[met] = sm_cnt
                id_to_sm_dict[sm_cnt] = met
                sm_cnt += 1
                met_cnt += 1
            if lig not in sm_to_id_dict:
                sm_to_id_dict[lig] = sm_cnt
                id_to_sm_dict[sm_cnt] = lig
                sm_cnt += 1
                lig_cnt += 1
            if sol not in sm_to_id_dict:
                sm_to_id_dict[sol] = sm_cnt
                id_to_sm_dict[sm_cnt] = sol
                sm_cnt += 1
                sol_cnt += 1
            if met == '':
                met = -1
            else:
                met = sm_to_id_dict[met]
            id_id_ee_list.append(
                (sm_to_id_dict[sm1], sm_to_id_dict[sm2], met, sm_to_id_dict[lig], sm_to_id_dict[sol], float(ee)))
        np.save(fname[:-4] + '_id.npy', id_id_ee_list)


sm_to_id_dict, id_to_sm_dict = dict(), dict()
sm_cnt, sm1_cnt, sm2_cnt, met_cnt, lig_cnt, sol_cnt = 0, 0, 0, 0, 0, 0

build_mol_dict(train_file)
build_mol_dict(test_file)

print(f'sm_cnt = {sm_cnt}, sm1_cnt = {sm1_cnt}, sm2_cnt = {sm2_cnt}, \
metal_cnt = {met_cnt}, ligand_cnt = {lig_cnt}, solvent_cnt = {sol_cnt}')

# save `sm_to_id_dict` to file
np.save(mol_id_dict_filename, sm_to_id_dict)
np.save(id_mol_dict_filename, id_to_sm_dict)


# dict = np.load(mol_id_dict_filename).item()

# use_metal_emb = False
# use_ligand_emb = True
# use_solvent_emb = False

def run_a_train_epoch(n_epochs, epoch, model, batched_mol_graphs, samples, loss_criterion, optimizer):
    model.train()
    # print(f'shape of samples: {np.array(samples).shape}')
    labels = torch.tensor(samples[:, -1]).float()
    samples = samples[:, :-1].astype(int)
    if torch.cuda.is_available():
        batched_mol_graphs.to(torch.device('cuda:0'))
    # g, samples, node_feats, edge_feats, get_node_weight=False
    prediction = model(batched_mol_graphs, samples, batched_mol_graphs.ndata['hv'],
                       batched_mol_graphs.edata['he']).squeeze()
    loss = (loss_criterion(prediction, labels).float()).mean()
    # loss = loss_criterion(prediction, labels)
    # print(loss.shape)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss = loss.data.item()
    # print(list(labels.size()),list(prediction.size()))
    mae = sum(np.abs(labels.detach().numpy() - prediction.detach().numpy())) / (list(labels.size())[0])
    print('epoch {:d}/{:d}, training total loss {:.4f}'.format(epoch + 1, n_epochs, total_loss))
    print('MAE: ', mae)
    return total_loss, mae


def test_model(batched_mol_graphs, samples):
    labels = torch.tensor(samples[:, -1]).float()
    samples = samples[:, :-1].astype(int)
    prediction = model(batched_mol_graphs, samples, batched_mol_graphs.ndata['hv'],
                       batched_mol_graphs.edata['he']).squeeze()
    mae = sum(np.abs(labels.detach().numpy() - prediction.detach().numpy())) / (list(labels.size())[0])
    print(f'---------- test MAE: {mae} -----------')
    return mae


# model = model_zoo.chem.AttentiveFP(node_feat_size=39,
#                                    # model = attentivefp.AttentiveFPGNN(node_feat_size=39,
#                                    edge_feat_size=10,
#                                    num_layers=2,
#                                    num_timesteps=2,
#                                    graph_feat_size=200,
#                                    output_size=1,
#                                    dropout=0.2)


# Load molecules
id_to_sm_dict = np.load(id_mol_dict_filename, allow_pickle=True).item()
all_mols = [id_to_sm_dict[i] for i in range(len(id_to_sm_dict))]

# set atom_featurizer and bond_featurizer
atom_featurizer = BaseAtomFeaturizer(
    {'hv': ConcatFeaturizer([
        partial(atom_type_one_hot, allowable_set=[
            'C', 'Br', 'N', 'O', 'Cl', 'F', 'P', 'S', 'I', 'Sn', 'Se', 'Si'
            , 'Ag', 'Au', 'Ni', 'Zn', 'Mg', 'Co', 'Fe', 'Mn', 'Cu', 'B', 'Sb'  # extra atom type from metal
        ], encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=list(range(6))),
        atom_formal_charge,
        atom_num_radical_electrons,
        partial(atom_hybridization_one_hot, encode_unknown=True),
        lambda atom: [0],  # A placeholder for aromatic information,
        atom_total_num_H_one_hot, chirality
    ]
    )})

# bond_featurizer = BaseBondFeaturizer({
#     'he': lambda bond: [0 for _ in range(10)]
# })

bond_featurizer = AttentiveFPBondFeaturizer(
    bond_data_field='he',
    self_loop=False  # self_loop = True
)

graph_list = [smiles_to_bigraph(mol,
                                node_featurizer=atom_featurizer,
                                edge_featurizer=bond_featurizer) for mol in all_mols]

atom_feat_size = list(graph_list[0].ndata['hv'][0].size())[0]
bond_feat_size = list(graph_list[0].edata['he'][0].size())[0]

print(f"atom_feat_size: {atom_feat_size}, bond_feat_size: {bond_feat_size}")

bg = dgl.batch(graph_list)
bg.set_n_initializer(dgl.init.zero_initializer)
bg.set_e_initializer(dgl.init.zero_initializer)

model = AFP_EE_Predictor(node_feat_size=atom_feat_size,
                         edge_feat_size=bond_feat_size,
                         num_layers=2,
                         num_timesteps=2,
                         graph_feat_size=200,
                         pred_input_size=800,  # 4*200 - no metal, 5*200 - metal
                         n_tasks=1,
                         dropout=0.
                         )

# load data
train_data = np.load(train_file[:-4] + '_id.npy', allow_pickle=True)
test_data = np.load(test_file[:-4] + '_id.npy', allow_pickle=True)

# set training regime
loss_fn = nn.MSELoss(reduction='none')
# loss_fn = nn.KLDivLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (-2.5), weight_decay=10 ** (-5.0))
n_epochs = 400
epochs, t_epos = [], []
scores, maes, tmaes = [], [], []
for e in range(n_epochs):
    score, mae = run_a_train_epoch(n_epochs, e, model, bg, train_data, loss_fn, optimizer)
    epochs.append(e)
    scores.append(score)
    if (e+1) % 10 == 0:
        tmae = test_model(bg, test_data)
        t_epos.append(e)
        tmaes.append(tmae)
    maes.append(mae)


torch.save(model, f'model_epo{n_epochs}.pkl')

plt.clf()
plt.plot(range(n_epochs), scores)
plt.title(f'Loss of training(epoch:{n_epochs})')
plt.savefig(f'loss_of_{n_epochs}.jpg')

plt.clf()
plt.plot(range(n_epochs), maes)
plt.title(f'MAE of training(epoch:{n_epochs})')
plt.savefig(f'MAE_{n_epochs}.jpg')

model.eval()

plt.clf()
plt.plot(t_epos, tmaes)
plt.title(f'MAE of testing (epoch:{n_epochs})')
plt.savefig(f'test MAE_{n_epochs}.jpg')

# Testing
print('MAE: ', test_model(bg, test_data))
