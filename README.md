# neural_manifolds
Tools to estimate low-dimensional latent space from high dimensional spiking data



## How to

You are not required to use a conda environment but we recommend it. To create a conda environment or the project with the required packages, run:

```bash
conda create -n neural_manifolds python=3.10
conda activate neural_manifolds
pip install requirements.txt
```

Install pre-commit hook:
```bash
pre-commit install
``` 


## Artificial data

To validate different dimensionality reduction methods, we generate artificial data.
Genrally we can divide the process into four steps. We create an underlying latent model (1). This model is than used to generate rates (2). From the rates we generate spikes (3). Finally, we subsample from the spikes to generate signals (4).

To see an general example of this see Notebook `notebooks/artificial_data.ipynb`.