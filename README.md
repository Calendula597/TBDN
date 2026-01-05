# TBDN
Official repository of "Think Bright, Diffuse Nice: Enhancing In-Context Learning via Inductive-Bias Hint Instruction and Query Contrastive Decoding"

# install torch
conda create -n tbdn python=3.10.12
conda activate tbdn

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# download dataset
Download the images and their corresponding descriptions of CoBSAT dataset.
wget "https://huggingface.co/datasets/yzeng58/CoBSAT/resolve/main/datasets.zip"

Uncompress the datasets.zip file via unzip datasets.zip and move the datasets folder into the cobsat folder.