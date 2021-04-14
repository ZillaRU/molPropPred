# Instruction
1. Create the env using conda.

    `conda install --yes --file requirements.txt`
    
2. Since the `dgl.chem` submodule has been moved to a single package `DGL-LifeSci` from `dgl` after v0.4.3, we need to alter the dgl package with its earlier version to make this code work.

    `conda install --use-local dgl_with_chem/xxxxx`