# BunDLe-Net
Behavioural and Dynamic Learning Network (BunDLe Net) is an algorithm to learn meaningful coarse-grained representations from time-series data. It maps high-dimensional data to low-dimensional space while preserving both dynamical and behavioural information. It has been applied, but is not limited, to neuronal manifold learning. 

After creating a new virtual environment, to install dependencies, you can run
```bash
python3 -m pip install -r requirements.txt
```
For the journal article on BunDLe-Net, see [https://www.biorxiv.org/content/10.1101/2023.08.08.551978v2](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v2)

For a demonstration of BunDLe-Net, please refer to the `main.py` file or explore the `notebooks/` directory for a Jupyter Notebook demo. You'll find all the heavy-lifting and the algorithm itself in the `functions/` directory.

For the comparision of BunDLe-Net with other commonly used embedding methods, and detailed evaluation of the embeddings, please see [https://github.com/akshey-kumar/comparison-algorithms](https://github.com/akshey-kumar/comparison-algorithms)

**BunDLe-Net embedding of *C.elegans* neuronal data in 3-D latent space**

![BunDLe-Net embedding of C.elegans neuronal data in 3-D latent space](https://github.com/akshey-kumar/BunDLe-Net/blob/main/figures/rotation_comparable_embeddings/rotation_BunDLeNet_worm_0.gif)
