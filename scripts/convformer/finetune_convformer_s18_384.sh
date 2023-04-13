DATA_PATH=/home/$USER/ba_project/datasets/test
CODE_PATH=/home/$USER/metaformer
INIT_CKPT=/home/$USER/metaformer/convformer_s18_384.pth

BATCH_SIZE=64
NUM_GPU=1
GRAD_ACCUM_STEPS=1

cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model convformer_s18_384 --img-size 384 --epochs 2 --opt adamw --lr 5e-5 --sched None \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--initial-checkpoint $INIT_CKPT \
--mixup 0 --cutmix 0 \
--model-ema --model-ema-decay 0.9999 \
--drop-path 0.3 --head-dropout 0.4 --input-size 3 224 224
