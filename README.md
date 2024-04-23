
![6vb3](./img/6vb3.png)

*Figure: Comparative Visualization of pMHC Complex Prediction Accuracy. True peptide structure in red, Pandora model in orange, MHC-Fine (fine-tuned AlphaFold) in blue. High precision of MHC-Fine (RMSD: 0.25 Å) versus Pandora (RMSD: 1.44 Å) for PDB ID: 6vb3.*

# MHC-Fine

The precise prediction of Major Histocompatibility Complex (MHC)-peptide complex structures is pivotal for understanding cellular immune responses and advancing vaccine design. In this study, we enhanced AlphaFold's capabilities by fine-tuning it with a specialized dataset comprised by exclusively high-resolution MHC-peptide crystal structures. This tailored approach aimed to address the generalist nature of AlphaFold's original training, which, while broad-ranging, lacked the granularity necessary for the high-precision demands of MHC-peptide interaction prediction. A comparative analysis was conducted against the homology-modeling-based method Pandora, as well as the AlphaFold multimer model. Our results demonstrate that our fine-tuned model outperforms both in terms of RMSD (median value is 0.65 Å) but also provides enhanced predicted lDDT scores, offering a more reliable assessment of the predicted structures. These advances have substantial implications for computational immunology, potentially accelerating the development of novel therapeutics and vaccines by providing a more precise computational lens through which to view MHC-peptide interactions.

# Table of Content
* [News](#markdown-header-news)
* [Pretrained Model](#markdown-header-pretrained-model)
* [Inference](#markdown-header-inference): [Colab](#markdown-header-remotely-using-colab), [Local](#markdown-header-locally-using-jupyter-notebook)
* [Citation](#markdown-header-citation)

# News

- **2024/04/23 Fast MSA Generation** The MSA for this protocol are generated using jackhmmer queries to a database generated for focusing on MHC complexes.

# Pretrained Model

You can download the latest version of the model from Google drive: [link](https://drive.google.com/file/d/1gz8uF8DKE0CzyX_WeDGOX7xP69LjpaZT/view?usp=sharing)

# Inference

To run inference, you need:

- protein sequence
- peptide sequence: length 8 to 11
- unique id: name for the sample

## Remotely using Colab

We prepared a Colab notebook which you can use:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1psEiqL2A4V28VwVKSlyx-FlHI15ZI-qs)

## Locally using Jupyter Notebook

If you want to run it locally, please follow the [instruction](#markdown-header-environment-setup-instruction) to set up the environment and 
run `jupyter notebook` to open [Inference.ipynb](./Inference.ipynb).

### Environment Setup Instruction

Option 1: using [mhc-fine.yml](mhc-fine.yml)

```
conda env create -f mhc-fine.yml
```

Option 2: 

If Option 1 does not work, please create the environment mannually with the following instructions.

```
conda create -n mhc-fine
conda activate mhc-fine
conda install libgcc-ng libstdcxx-ng -c conda-forge
conda install kalign3=3.4 -c bioconda
# If you have a different cuda version, please visit pytorch website
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install biopython gdown -c conda-forge
conda install absl-py scipy pandas jupyter -c anaconda
```

# Citation

```
@article{Glukhov2023MHCFine,
  title={MHC-Fine: Fine-tuned AlphaFold for Precise MHC-Peptide Complex Prediction},
  author={Ernest Glukhov and Dmytro Kalitin and Darya Stepanenko and Yimin Zhu and Thu Nguen and George Jones and Carlos Simmerling and Julie C. Mitchell and Sandor Vajda and Ken A. Dill and Dzmitry Padhorny and Dima Kozakov},
  journal={bioRxiv},
  year={2023},
}
```