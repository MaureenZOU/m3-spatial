CUDA_VISIBLE_DEVICES=1 python train.py \
--source_path /disk1/data/m3/data_v2/train \
--model_path /home/xueyan/output/mmm/ckpt/test \
--postfix "test" \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--bsz 1 \
--eval \
--use_embed \
--use_llama3 \
--use_clip \
--use_siglip 
# --use_seem \
# --use_dinov2 \
# --use_llamav \
# --use_wandb