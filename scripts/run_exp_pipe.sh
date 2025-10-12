#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

######### check number of args ##########
HELP="Usage example: GPU=0 bash $0 <name> <AE config> <LDM pretrain config> <LDM config> <interface encoder config> [mode: e.g. 1111]"
echo ${HELP}

# default
NAME=PeptideMimicry
AECONFIG=${CODE_DIR}/configs/train_autoencoder.yaml
PreLDMCONFIG=${CODE_DIR}/configs/pretrain_ldm.yaml
LDMCONFIG=${CODE_DIR}/configs/train_ldm.yaml
IFCONFIG=${CODE_DIR}/configs/train_ifencoder.yaml

if [ -z $1 ]; then
    echo "Experiment name using default: ${NAME}."
else
    NAME=$1
fi
if [ -z $2 ]; then
    echo "Autoencoder config using default: ${AECONFIG}."
else
    AECONFIG=$2
fi
if [ -z $3 ]; then
    echo "LDM pretrain config using default: ${PreLDMCONFIG}"
else
    PreLDMCONFIG=$3
fi
if [ -z $4 ]; then
    echo "LDM config using default: ${LDMCONFIG}"
else
    LDMCONFIG=$4
fi
if [ -z $5 ]; then
    echo "Interface encoder config using default: ${IFCONFIG}."
else
    IFCONFIG=$5
fi

if [ -z $6 ]; then
    MODE=1111
else
    MODE=$6
fi
echo "Mode: $MODE, [train AE] / [pretrain LDM] / [train LDM] / [train Interface Encoder]"
TRAIN_AE_FLAG=${MODE:0:1}
PRETRAIN_LDM_FLAG=${MODE:1:1}
TRAIN_LDM_FLAG=${MODE:2:1}
TRAIN_IFENC_FLAG=${MODE:3:1}

AE_SAVE_DIR=./exps/$NAME/AE
PRE_LDM_SAVE_DIR=./exps/$NAME/PreLDM
LDM_SAVE_DIR=./exps/$NAME/LDM
IFENC_SAVE_DIR=./exps/$NAME/IFEncoder
OUTLOG=./exps/$NAME/output.log

if [[ ! -e ./exps/$NAME ]]; then
    mkdir -p ./exps/$NAME
elif [[ -e $AE_SAVE_DIR ]] && [ "$TRAIN_AE_FLAG" = "1" ]; then
    echo "Directory ${AE_SAVE_DIR} exisits! But training flag is 1!"
    exit 1;
elif [[ -e $PRE_LDM_SAVE_DIR ]] && [ "$PRETRAIN_LDM_FLAG" = "1" ]; then
    echo "Directory ${PRE_LDM_SAVE_DIR} exisits! But training flag is 1!"
    exit 1;
elif [[ -e $LDM_SAVE_DIR ]] && [ "$TRAIN_LDM_FLAG" = "1" ]; then
    echo "Directory ${LDM_SAVE_DIR} exisits! But training flag is 1!"
    exit 1;
elif [[ -e $IFENC_SAVE_DIR ]] && [ "$TRAIN_IFENC_FLAG" = "1" ]; then
    echo "Directory ${IFENC_SAVE_DIR} exisits! But training flag is 1!"
    exit 1;
fi

########## train autoencoder ##########
echo "Training Autoencoder with config $AECONFIG:" | tee $OUTLOG
cat $AECONFIG | tee -a $OUTLOG
if [ "$TRAIN_AE_FLAG" = "1" ]; then
    bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR
fi
echo "" | tee -a $OUTLOG

########## pretrain ldm ##########
echo "Pretraining LDM with config $PreLDMCONFIG:" | tee -a $OUTLOG
cat $PreLDMCONFIG | tee -a $OUTLOG
AE_CKPT=`cat ${AE_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using Autoencoder checkpoint: ${AE_CKPT}" | tee -a $OUTLOG
if [ "$PRETRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $PreLDMCONFIG --trainer.config.save_dir=$PRE_LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT
fi
echo "" | tee -a $OUTLOG

########## train ldm ##########
echo "Training LDM with config $LDMCONFIG:" | tee -a $OUTLOG
cat $LDMCONFIG | tee -a $OUTLOG
PRELDM_CKPT=`cat ${PRE_LDM_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using pretrained checkpoint: ${PRELDM_CKPT}" | tee -a $OUTLOG
if [ "$TRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT --load_ckpt=$PRELDM_CKPT
fi
echo "" | tee -a $OUTLOG

########## get latent distance ##########
LDM_CKPT=`cat ${LDM_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Get distances in latent space" | tee -a $OUTLOG
python setup_latent_guidance.py --config configs/setup_latent_guidance.yaml --ckpt ${LDM_CKPT} --gpu ${GPU:0:1} >> $OUTLOG
echo "" | tee -a $OUTLOG


########## train interface encoder ##########
echo "Training Interface Encoder with config ${IFCONFIG}:" | tee -a $OUTLOG
cat $IFCONFIG | tee -a $OUTLOG
echo "Using LDM checkpoint for training interface encoder: ${LDM_CKPT}" | tee -a $OUTLOG
if [ "$TRAIN_IFENC_FLAG" = "1" ]; then
    bash scripts/train.sh $IFCONFIG --trainer.config.save_dir=$IFENC_SAVE_DIR --model.ldm_ckpt=$LDM_CKPT >> $OUTLOG
fi
echo "" | tee -a $OUTLOG

# get final checkpoint
IFENC_CKPT=`cat ${IFENC_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Final peptide mimicry checkpoint: ${IFENC_CKPT}" | tee -a $OUTLOG
cp $IFENC_CKPT ./exps/$NAME/model.ckpt
echo "Copied final checkpoint to ./exps/$NAME/model.ckpt" | tee -a $OUTLOG