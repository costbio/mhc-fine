
![6vb3](./img/6vb3.png)

*Figure: Comparative Visualization of pMHC Complex Prediction Accuracy. True peptide structure in red, Pandora model in orange, MHC-Fine (fine-tuned AlphaFold) in blue. High precision of MHC-Fine (RMSD: 0.25 Å) versus Pandora (RMSD: 1.44 Å) for PDB ID: 6vb3.*

# MHC-Fine

The precise prediction of Major Histocompatibility Complex (MHC)-peptide complex structures is pivotal for understanding cellular immune responses and advancing vaccine design. In this study, we enhanced AlphaFold's capabilities by fine-tuning it with a specialized dataset comprised by exclusively high-resolution MHC-peptide crystal structures. This tailored approach aimed to address the generalist nature of AlphaFold's original training, which, while broad-ranging, lacked the granularity necessary for the high-precision demands of MHC-peptide interaction prediction. A comparative analysis was conducted against the homology-modeling-based method Pandora, as well as the AlphaFold multimer model. Our results demonstrate that our fine-tuned model outperforms both in terms of RMSD (median value is 0.65 Å) but also provides enhanced predicted lDDT scores, offering a more reliable assessment of the predicted structures. These advances have substantial implications for computational immunology, potentially accelerating the development of novel therapeutics and vaccines by providing a more precise computational lens through which to view MHC-peptide interactions.

# Pretrained Model

You can download the latest version of the model from Google drive: [link](https://drive.google.com/file/d/1gz8uF8DKE0CzyX_WeDGOX7xP69LjpaZT/view?usp=sharing)

# Environment Setup

## Create virtual environment

`conda create -n mhc-fine python=3.12`

`conda activate mhc-fine`

## Install libraries

Check cuda version to select corresponding pytorch version: `nvidia-smi`

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

`pip install Bio`

`pip install notebook`

`pip install absl-py`

`pip install scipy`

`pip install gdown`

`conda install https://anaconda.org/bioconda/kalign3/3.4.0/download/linux-64/kalign3-3.4.0-hdbdd923_0.tar.bz2`

# Inference

## Locally using Jupyter Notebook

`jupyter notebook`

Open [Inference.ipynb](./Inference.ipynb).

## Remotely using Colab

To run inference, you need:

- protein sequence
- peptide sequence: length 8 to 11
- unique id: name for the sample

We prepared a Colab notebook which you can use:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1psEiqL2A4V28VwVKSlyx-FlHI15ZI-qs)