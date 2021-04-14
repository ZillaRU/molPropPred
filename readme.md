# Instruction
1. create the env by conda

    `conda install --yes --file requirements.txt`
    
2. Since the `dgl.chem` submodule has been moved to a single package `DGL-LifeSci` from `dgl` after v0.4.3, we need to alter the dgl with a earlier version to make this code work.