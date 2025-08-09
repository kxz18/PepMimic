# Peptide Mimicry

![cover](./assets/cover.png)

## Environment

:warning: The codes are tested under cuda 11.7.

```bash
conda env create -f env.yaml
conda activate pepmimic
```

## Checkpoints

The model weights can be downloaded at the [release page](https://github.com/kxz18/PepMimic/releases/download/v1.0/checkpoints.zip).

```bash
wget https://github.com/kxz18/PepMimic/releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip
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
