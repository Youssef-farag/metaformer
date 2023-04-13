DATA_PATH=/home/yous/Desktop/cerrion/datasets/incremental_dataset_sections_conveyor_curve/
CODE_PATH=/home/yous/Desktop/cerrion/metaformer # modify code path here
INIT_CKPT=/home/yous/Desktop/cerrion/metaformer/convformer_s18_384.pth

BATCH_SIZE=2
NUM_GPU=1
GRAD_ACCUM_STEPS=1 # Adjust according to your GPU numbers and memory size.
#BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model convformer_s18_384 --img-size 384 --epochs 30 --opt adamw --lr 5e-5 --sched None \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--initial-checkpoint $INIT_CKPT \
--mixup 0 --cutmix 0 \
--model-ema --model-ema-decay 0.9999 \
--drop-path 0.3 --head-dropout 0.4