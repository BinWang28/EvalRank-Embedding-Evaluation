


# create a new virtual environment with conda (without asking confirmation)
conda create -n evalrank_env python=3.8 --yes

# activate
conda activate evalrank_env

# word evaluation essentials
conda install -c anaconda scikit-learn --yes
conda install -c conda-forge tqdm --yes
conda install -c conda-forge prettytable --yes


# Pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install -c conda-forge python-dateutil

# install transformers

pip install httplib2
pip install transformers==4.11.3