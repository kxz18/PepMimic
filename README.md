# Peptide Mimicry

## Environment

:warning: The codes are tested under cuda 11.7.

```bash
conda env create -f env.yaml
conda activate pepmimic
```


## Reproducing the Figures of In Silico Evaluations

First download the raw structure data for these figures from [google drive](https://drive.google.com/file/d/1QVevyLjq9Z66RoS6l9XAFgx6lOELKRut/view?usp=sharing) and decompress them under `figures/`:

```bash
figures/
├── codes
├── data
│   ├── test_metric
│   └── traj
└── draw.sh
```

Then run the following integrated script:

```bash
bash figures/draw.sh
```

## Mimicking Given References

:warning: You need to manually setup FoldX5 suite by acquiring an [academic license](https://foldxsuite.crg.eu/academic-license-info). The suite should be downloaded and extracted under `evaluation/dG/foldx5`:

```bash
evaluation/dG/foldx5/
├── foldx_20251231
├── molecules
└── yasaraPlugin.zip
```

The suffix "20251231" denotes the last valid day of usage (2025/12/31) since foldx only provide 1-year license for academic usage and thus needs yearly renewal. After renewal, the path in `globals.py` also needs to be changed according to the new suffix.

Prepare an input folder with reference complexes in PDB format, an index file containing the chain ids of the target protein and the ligand, as well as a configuration indicating parameters like peptide length and number of generations. We have prepared an example folder under `example_data/CD38`:

```bash
example_data/
└── CD38
    ├── 4cmh.pdb
    ├── 5f1o.pdb
    ├── config.yaml
    └── index.txt
```

You can try the generation with the following script:

```bash
# The last number 10 indicates we will finally select the best 10 candidates as the output
GPU=0 bash scripts/mimic.sh example_data/CD38 10
```

The results will be saved under `example_data/CD38/final_output`.

## Training

First download the datasets from [google drive](https://drive.google.com/file/d/1AC6d6eG5T-31_vZUi_d416owW3jSQe32/view?usp=sharing) and decompress them:

```bash
datasets
├── train_valid     # training and validation sets
├── LNR             # test set
└── ProtFrag        # augmentation data from protein monomers for pretraining
```

Then run the integrated scripts:

```bash
GPU=0 bash scripts/run_exp_pipe.sh
```

The resulting checkpoint will be saved at `./exps/PeptideMimicry/model.ckpt`.
