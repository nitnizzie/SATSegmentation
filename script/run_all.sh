############# run in single GPU ##############
GPUS=(0)
NUM_GPUS=1
##############################################
i=0
#LR="1e-3"

LR="1e-3 5e-4 1e-4 5e-5 1e-5"
EPOCH="100 200 400"

for lr in $LR; do
  for epoch in $EPOCH; do
    python train.py \
          --gpu_idx 0 \
          -m DeepLabV3 \
          --preprocess_fn \
          --transform 2 \
          --loss_fn dice \
          -lr "$lr" \
          -ep "$epoch" \
          --home '/content/drive/MyDrive/Dacon' \
          2>&1 &
  done
done
