#!/bin/bash
HELP="Usage example: GPU=0 bash $0 <directory> <topk>"
if [ -z $1 ]; then
    echo "Reference directory missing. ${HELP}"
    exit 1;
fi
if [ -z $2 ]; then
    echo "TopK not specified. ${HELP}"
    exit 1;
fi
if [ -z $GPU ]; then
    echo "GPU not specified. Using default: 0"
    GPU=0
fi

DATA_DIR=$1
TOPK=$2
export CUDA_VISIBLE_DEVICES=$GPU
NUM_CPU=8  # You can adjust number of CPUs to use here (for rosetta, foldx, and interface hit)

########## Generation of Peptide Mimicries ##########
echo "Step 1: Generating peptide mimicries..."
python mimic_design.py --config ${DATA_DIR}/config.yaml --gpu 0
echo "Step 1 finished"


########## Rosetta Evaluation ##########
echo "Step 2 (Rosetta): Calculating Rosetta energy..."
python -m evaluation.runner.pyrosetta_dG --root_dir ${DATA_DIR} --n_cpus ${NUM_CPU}
echo "Step 2 (Rosetta) finished"


########## FoldX Evaluation ##########
echo "Step 2 (FoldX): Calculating FoldX energy..."
python -m evaluation.runner.foldx_dG --root_dir ${DATA_DIR} --n_cpus ${NUM_CPU}
echo "Step 2 (FoldX) finished"


########## Interface Hit Evaluation ##########
echo "Step 2 (Interface Hit): Calculating Interface Hit..."
python -m evaluation.runner.interface_hit --root_dir ${DATA_DIR} --n_cpus ${NUM_CPU}
echo "Step 2 (Interface Hit) finished"


########## Selection of Final Candidates ##########
echo "Step 3: Selecting TopK..."
python -m evaluation.analyzer.select_final --root_dir ${DATA_DIR} --filter_name pyrosetta_dG foldx_dG --size ${TOPK}
echo "All Finished!"