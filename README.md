# Peptide Mimicry

![cover](./assets/cover.png)

## Environment

:warning: The codes are tested under cuda 11.7.

```bash
conda env create -f env.yaml
conda activate pepmimic
```

:warning: You need to further manually setup FoldX5 suite by acquiring an [academic license](https://foldxsuite.crg.eu/academic-license-info). The suite should be downloaded and extracted under `evaluation/dG/foldx5`:

```bash
evaluation/dG/foldx5/
├── foldx_20251231
├── molecules
└── yasaraPlugin.zip
```

The suffix "20251231" denotes the last valid day of usage (2025/12/31) since foldx only provide 1-year license for academic usage and thus needs yearly renewal. After renewal, the path in `globals.py` also needs to be changed according to the new suffix.

## Checkpoints

The model weights can be downloaded at the [release page](https://github.com/kxz18/PepMimic/releases/download/v1.0/checkpoints.zip).

```bash
wget https://github.com/kxz18/PepMimic/releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip
```

## Mimicking Given References

Prepare an input folder with reference complexes in PDB format, an index file containing the chain ids of the target protein and the ligand, as well as a configuration indicating parameters like peptide length and number of generations. We have prepared an example folder under `example_data/CD38`:

```bash
example_data/
└── CD38
    ├── 4cmh.pdb
    ├── 5f1o.pdb
    ├── config.yaml
    └── index.txt
```

Here we also illustrate the meaning of each entry in the `config.yaml`:

```yaml
dataset:
  test:
    class: MimicryDataset
    ref_dir: ./example_data/CD38    # The directory for all reference complexes, which should be a relative path rooted at the project folder, or a absolute path
    n_sample_per_cplx: 20           # The number of generations for each reference complex. This is just a toy example for a quick tour. For practical usage, we recommend generating a total of above 100,000 candidates before ranking to select the top-scoring one for wetlab tests. For example, here we have two reference complexes, thus we should set n_sample_per_cplx to at least 50,000, so that the total generations will be above 100,000.
    length_lb: 10                 # lower bound of peptide length (inclusive)
    length_ub: 12                 # uppper bound of peptide length (inclusive)

dataloader:
  num_workers: 4                  # Number of CPUs for data processing. Usually 4 is enough.
  batch_size: 32                  # If the GPU is out of memory, please try to reduce the batch size
```

Each line of `index.txt` describes the filename (without `.pdb`), the target chains, the reference ligand chains, and custom annotations for a reference complex, separated by `\t`. For example, the line for `4cmh.pdb` looks like:

```
4cmh	A	B,C	HEAVY CHAIN OF SAR650984-FAB FRAGMENT,LIGHT CHAIN OF SAR650984-FAB FRAGMENT
```

After preparation of these input files, you can try the generation with the following script:

```bash
# The last number 10 indicates we will finally select the best 10 candidates as the output
GPU=0 bash scripts/mimic.sh example_data/CD38 10
```

The results will be saved under `example_data/CD38/final_output`.
