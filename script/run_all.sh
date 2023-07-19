############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0
#LR="1e-3"

LR="1e-3 5e-4 1e-4 5e-5 1e-5"
EPOCH="100 200 400"



wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=4 #12
  local num_max_jobs=4
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

for lr in $LR; do
  for epoch in $EPOCH; do
    python train.py \
          --gpu_idx ${GPUS[$i]} \
          -m DeepLabV3 \
          --preprocess_fn \
          -lr "$lr" \
          -ep "$epoch" \
          2>&1 &
    wait_n
    # increment i
    ((i=i+1))

    python train.py \
          --gpu_idx ${GPUS[$i]} \
          -m DeepLabV3 \
          --preprocess_fn \
          --loss_fn dice \
          -lr "$lr" \
          -ep "$epoch" \
          2>&1 &
    wait_n
    # increment i
    ((i=i+1))
  #
    python train.py \
          --gpu_idx ${GPUS[$i]} \
          -m DeepLabV3 \
          --loss_fn dice_v2 \
          --preprocess_fn \
          -lr "$lr" \
          -ep "$epoch" \
          2>&1 &
    wait_n
    # increment i
    ((i=i+1))
  done
done
