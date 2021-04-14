# Instruction
1. Create the env using conda.

    `conda install --yes --file requirements.txt`
    
2. Since the `dgl.chem` submodule has been moved to a single package `DGL-LifeSci` from `dgl` after v0.4.3, we need to alter the dgl package with its earlier version to make this code work.

    `conda install --use-local dgl_with_chem/xxxxx`
    
# Caution
You may confuse for the error `OSError: [WinError 126]`
Possible causes:
1. missing runtime c++ library
2. some package(s) are incompatible for version gap

After fixing the bug above by using `pip install dgl==0.4.2` instead of `conda install ...`, I got another error prompt `'rdmolfiles' is not defined`.
I looked into the source code of `mol_to_graph()` and eventually realized that an importError occurred soundlessly.
![After installing the missing `matraj`, then everything will be OK](微信截图_20210414160801.png)