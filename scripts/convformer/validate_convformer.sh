DATA_PATH=/home/$USER/ba_project/datasets/retest
CODE_PATH=/home/$USER/metaformer
INIT_CKPT=/home/$USER/metaformer/convformer_s18_384.pth

BATCH_SIZE=64
NUM_GPU=1

cd $CODE_PATH && sh distributed_validate.sh $NUM_GPU $DATA_PATH \
--model convformer_s18_384 -b $BATCH_SIZE --checkpoint $INIT_CKPT
