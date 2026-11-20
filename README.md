# two-sprite-compositional-learning
## Overview

This project investigates how different data sampling strategies influence compositional generalisation and learned representations in neural networks.

Using the **dSprites dataset**, we construct scenes containing two sprites and study how the distribution of training samples affects a model's ability to generalise to unseen combinations of object attributes.

## Research Question

**How does the sampling strategy used to construct a two-sprite training dataset influence compositional generalisation and the structure of learned representations?**

## Research Objectives

The project has two main objectives:

**1. Compositional Generalisation**
- Evaluate performance on in-distribution (ID) samples.
- Test generalisation to held-out combinations (OOD).
- Evaluate generalisation to continuously sampled latent configurations.

**2. Representation Analysis**
- Measure factorisation and orthogonality of learned representations.
- Evaluate attribute information using linear probes.
- Investigate cross-slot information leakage between the two sprites.

## Methodology

We compare different training data sampling strategies:

- Random sampling (baseline).
- Orthogonal sampling.
- Diagonal sampling.
- Structured or block sampling.

To ensure fair comparisons, we keep the model architecture, loss function, optimizer, training schedule, and evaluation datasets fixed. Experiments are repeated across multiple random seeds.

## Model

The initial model consists of two independent neural network branches, each generating a sprite image from its latent attributes. The generated images are combined using an occlusion-based composition function to produce the final scene.

## Repository Structure

```text
two-sprite-compositional-learning/
├── main.py
├── src/
│   ├── train.py
│   ├── model.py
│   ├── data.py
│   └── utils.py
├── cfgs/
├── experiments/
├── requirements.txt
└── README.md
```



## Data

### Dataset paths

The dataset path is configured in `main.py`. Update this path to match your local setup before running any experiments.

### Datasets

- **dSprites**: Download the dataset [here](YOUR_DATASET_DOWNLOAD_LINK).

After downloading, place the dataset in a local directory and update the dataset root path in `main.py`.

For example:

```python
"data": {"root": "/path/to/your/dsprites"}
```

The dataset is not included in this GitHub repository and must be downloaded separately.










## Installation and Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default experiment:

```bash
python main.py
```

Run an experiment with a custom configuration:

```bash
python main.py --config cfgs/baseline.yml
```


## Representation Extraction and Analysis

This project includes scripts for extracting and analysing internal representations from trained models. The analysis is used to investigate how different training data sampling strategies and model architectures influence the geometry and structure of learned representations.

### 1. Representation Extraction

After training a model, extract its internal representations using:

```bash
python src/extract_representations.py --help
```

The extraction script supports loading latent configurations from `.npy` files using the `--latents` argument. This allows representations to be extracted from different datasets, including held-out evaluation sets for out-of-distribution (OOD) analysis.

Representations are extracted at three levels of the network:

- **Initial**: Output of the final fully connected layer, before unflattening.
- **Intermediate**: Output of the first convolutional layer.
- **Final**: Output of the penultimate convolutional layer, before the final image-producing convolution.

The extracted representations and corresponding latent labels are saved in the `representations/` directory.

### 2. Representation Analysis

Two scripts are available for analysing the extracted representations.

**Standard analysis:**

```bash
python src/analyze_representations.py --help
```

**GPU-accelerated analysis:**

```bash
python src/analyze_representations_gpu.py --help
```

The GPU version uses CUDA when available to accelerate linear probe fitting.

The analysis evaluates the following properties of the learned representations:

- **Linear probes**: Measure how accurately individual generative factors can be decoded using linear classifiers.
- **Factorisation**: Evaluate how well the representation separates information associated with individual generative factors.
- **Orthogonality**: Measure cosine similarities between factor-associated directions.
- **Principal angles**: Measure geometric relationships between factor subspaces.
- **Dimensionality**: Estimate the effective rank and the number of dimensions required to explain 95% of the variance.
- **Cross-slot leakage**: Measure how much information about one sprite's attributes is present in the representation of the other sprite. This analysis applies to compositional models.

### 3. Analysis Results

Analysis outputs are saved in the `analysis_results_gpu/` directory when using the GPU analysis script.

The directory is organised by experimental condition, representation level, and model architecture:

```text
analysis_results_gpu/
├── random_compositional/
├── orthogonal_compositional/
├── diagonal_compositional/
├── random_monolithic/
├── orthogonal_monolithic/
└── diagonal_monolithic/
```

Each experimental condition contains separate directories for the initial, intermediate, and final representation levels.

Compositional models have separate representation directories for each sprite slot (`slot1` and `slot2`), whereas monolithic models use a shared representation directory (`shared`).

The analysis exports CSV files containing the calculated metrics:

- `linear_probes.csv`
- `factorisation.csv`
- `factor_dimensionality.csv`
- `factor_subspace_overlap.csv`
- `orthogonality_matrix.csv`
- `orthogonality_pairs.csv`
- `orthogonality_summary.csv`
- `overall_dimensionality.csv`
- `cross_slot_leakage.csv` (compositional models only)

These files can be used for further analysis and visualisation in Python notebooks.

### 4. Reproducibility

For meaningful comparisons across experimental conditions, use the same latent evaluation samples and their ordering when extracting representations from different checkpoints.

Record the checkpoint, experimental condition, representation level, and analysis settings to ensure reproducibility.

## Goal

The goal is to understand how training data distributions shape learned representations and influence a neural network's ability to generalise compositionally to unseen configurations.


![Two-Sprite Compositional Learning](images/DS.gif)



## References

- [Necessary Compositionality](https://arxiv.org/pdf/2602.24264)
- [Necessary Compositionality — Code](https://github.com/oshapio/necessary-compositionality/tree/main/src/complinearity)
- [Compositional OOD Generalization](https://github.com/brendel-group/compositional-ood-generalization)
