# CSRec
Contrastive Learning Social Recommendation

We propose a new social recommendation method, which leaverage the social information to enhance the user representation and contrastive learning to refine the soical graph.

# Contrastive Learning
[Self-supervised Graph Learning for Recommendation](https://arxiv.org/abs/2010.10783)

# Dataset
We provide two datasets: [LastFM](https://grouplens.org/datasets/hetrec-2011/) and [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm).

# Example to run the codes
1. Environment: Please install Python 3.6, PyTorch 1.4.0, and other requirements as follows:
    `pip install -r requirements.txt`
2. Run **CSRec** on the LastFM dataset:

    `python main.py --model=CSRec --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048`
