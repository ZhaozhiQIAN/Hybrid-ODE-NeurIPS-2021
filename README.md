# Hybrid-ODE-NeurIPS-2021
Code for [Integrating Expert ODEs into Neural ODEs: Pharmacology and Disease Progression (NeurIPS 2021)](https://papers.neurips.cc/paper/2021/hash/5ea1649a31336092c05438df996a3e59-Abstract.html).



## Installation

Python 3.6+ is recommended. Install dependencies as per [`requirements.txt`](./requirements.txt). 

If [CUDA](https://developer.nvidia.com/cuda-zone) support is needed: 
* make sure you have [appropriate drivers](https://www.nvidia.co.uk/Download/index.aspx) installed, 
* make sure you have [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (a version compatible with PyTorch `1.10`, see [here](https://pytorch.org/get-started/locally/) or [here](https://pytorch.org/get-started/previous-versions/)) installed on your system or in you virtual environment.



## Replicating Experiments

Shell scripts to replicate the experiments can be found in [`experiments/`](./experiments/).

To run all the synthetic data experiments:
```bash
$ bash experiments/run_all.sh
```
You may also run the experiment steps individually, see [`experiments/run_all.sh`](./experiments/run_all.sh). To then produce the figures, run the Jupyter notebooks `Fig3.ipynb`, `Fig6.ipynb`, `Fig7.ipynb`, `Fig9.ipynb` found under [`experiments/`](./experiments/).

To run real data experiments (access to Dutch Data Warehouse dataset is required, see [`real_data/README.md`](./real_data/README.md) for more information):
```bash
$ bash experiments/real.sh
```



## Citing

If you use this code, please cite the associated paper:

```
@inproceedings{NEURIPS2021,
  author = {Qian, Zhaozhi and Zame, William R and Fleuren, Lucas M and Elbers, Paul and van der Schaar, Mihaela},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Integrating Expert ODEs into Neural ODEs: Pharmacology and Disease Progression},
  url = {https://papers.neurips.cc/paper/2021/file/5ea1649a31336092c05438df996a3e59-Paper.pdf},
  volume = {34},
  year = {2021}
}
```
