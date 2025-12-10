# Morph: Decentralized Learning with Dissimilarity-Driven Peer Selection

This repository contains code for a decentralized learning framework
where nodes adaptively select peers based on model and data
dissimilarity. The project includes infrastructure for training,
evaluating, and analyzing behavior across various topologies and
datasets.

## Setup

-   Fork the repository.

-   Clone and enter your local repository.

-   Check if you have Python \>= 3.8:

    ``` bash
    python --version
    ```

-   (Optional) Create and activate a virtual environment:

    ``` bash
    python3 -m venv [venv-name]
    source [venv-name]/bin/activate
    ```

-   Upgrade pip:

    ``` bash
    pip3 install --upgrade pip
    ```

-   On Mac M1, `pyzmq` may fail with pip — install it via conda:
    <https://conda.io>

-   Install decentralizepy in development mode:

    ``` bash
    # zsh
    pip3 install --editable .\[dev\]

    # bash
    pip3 install --editable .[dev]
    ```

-   Install psutil:

        pip3 install psutil

-   Download CIFAR-10:

    ``` bash
    python download_dataset.py
    ```

-   Download FEMNIST from the [LEAF
    repository](https://github.com/TalwalkarLab/leaf) and place it in:

    ``` bash
    eval/data/femnist/
    ```

## Directory Structure

``` text
src/decentralizepy/nodes/DissDL/
├── DissDL.py                  # Main training algorithm
├── GlobalModelSimilarity.py   # Computes cosine similarity to global model
├── GlobalDataSimilarity.py    # Tracks similarity to global data distribution

tutorial/DissDL/
├── run.sh                     # Entry point for all experiments
├── config.ini                 # CIFAR-10 configuration
├── config_fem.ini             # FEMNIST configuration
├── testingDissDL.py           # Switches between algorithm/experiment types

StaticTopologies/              # Static baselines (same structure as DissDL)
EpidemicLearning/              # EL-Local implementation (same structure)

data/experiments/
├── cifar/
│   ├── degree_3/
│   │   ├── diss_dl/
│   │   ├── static_mh/
│   │   ├── epidemic/
│   │   └── fully_connected/
│   └── degree_7/
│       ├── diss_dl/
│       ├── static_mh/
│       ├── epidemic/
│       └── fully_connected/
├── femnist/
│   └── degree_3/
│       ├── diss_dl/
│       ├── static_mh/
│       ├── epidemic/
│       └── fully_connected/
├── similarity/
│   ├── model/
│   │   ├── cifar_3/
│   │   └── cifar_7/
│   └── data_distribution/
│       ├── cifar_3/
│       └── cifar_7/
```

## Datasets

-   **CIFAR-10** Used in topologies of degree 3 and 7.
-   **FEMNIST** Partitioned across users; obtained from the [LEAF
    benchmark](https://github.com/TalwalkarLab/leaf).

## Configuration

Config files specify experiment parameters:

-   `config.ini`: for CIFAR-10
-   `config_fem.ini`: for FEMNIST

You can modify:

-   Topology degree
-   Random seed
-   Algorithm (via file in `testingDissDL.py`)
-   Training parameters (e.g., batch size, learning rate)

## Running Experiments

Navigate to the experiment directory and run:

``` bash
cd tutorial/DissDL
./run.sh
```

Inside `run.sh`, you can:

-   Change the **graph topology**
-   Choose the **configuration** (FEMNIST or CIFAR-10)
-   Select the algorithm/experiment via `testingDissDL.py`:
    -   `DissDL.py` – standard training
    -   `GlobalModelSimilarity.py` – model similarity
    -   `GlobalDataSimilarity.py` – data distribution similarity

## Experimental Types

1.  **Main Algorithm Evaluation**

    Evaluates algorithm performance across topologies and seeds.

    ``` text
    data/experiments/{dataset}/{degree}/{algorithm}/run_{n}/
    ```

2.  **Similarity Experiments**

    -   **Global Model Similarity**: Tracks cosine similarity between
        local and global models over time.
    -   **Global Data Distribution Similarity**: Tracks how closely a
        node's aggregated data approximates the true global
        distribution.

    ``` text
    data/experiments/similarity/{experiment_type}/{dataset}/run_{n}/
    ```

    where `{experiment_type}` is either `model` or `data_distribution`.

## Results & Plotting

The `eval/` directory contains scripts for aggregating results and
generating plots.

1.  **Main Algorithm Evaluation**

    -   `evaluation.py`: Aggregates metrics from multiple runs into a
        CSV summary file.

        Folder path can be modified near the bottom of the script to
        point to a specific experiment:

        ``` text
        data/experiments/{dataset}/{degree}/{algorithm}
        ```

    -   `plot_inter_run_cifar.py` / `plot_inter_run_femnist.py`: Plot
        test accuracy, loss, inter-node variance, and communication cost
        over time.

        Update the input folder path at the top of the script to switch
        experiments.

2.  **Similarity Experiments**

    -   `plot_model_similarity.py`: Visualizes how cosine similarity
        between each node’s model and the global model evolves through
        rounds.
    -   `plot_data_similarity.py`: Visualizes how closely each node’s
        aggregated data distribution matches the global data
        distribution.

    In both cases, set the appropriate folder at the bottom of the
    script:

    ``` text
    data/experiments/similarity/{experiment_type}/{dataset}
    ```

    where `{experiment_type}` is either `model` or `data_distribution`.

# Enabling GPU Usage

To enable GPU acceleration, add the following to your configuration file
(by default it is false):

``` ini
[CUDA]
use_cuda = true
```

Make sure your system has a compatible CUDA-enabled GPU and that
`torch.cuda.is_available()` returns `True`.

This setting ensures the model and training tensors are moved to the GPU
when available.
