# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#   --nproc_per_node=2 \
#   --master_port=29988 \
#   test.py \
#   --launcher pytorch \
#   --tcp_port 29988 \
#   --cfg_file ./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
#   --ckpt ../weights/checkpoint_epoch_36_nus_mamba.pth \
#   --batch_size 2 \
#   --workers 4 
#   # --extra_tag nusc_mini \
#   # --eval_tag mini \
#   --set DATA_CONFIG.VERSION v1.0-trainval

# CUDA_VISIBLE_DEVICES=0 python test.py \
#   --cfg_file ./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
#   --ckpt ../weights/checkpoint_epoch_36_nus_mamba.pth \
#   --batch_size 2 \
#   --workers 4 \
#   --set DATA_CONFIG.VERSION v1.0-trainval

CUDA_VISIBLE_DEVICES=0,1 TORCH_CUDNN_V8_API_ENABLED=0 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29988 \
  test.py \
  --launcher pytorch \
  --tcp_port 29988 \
  --cfg_file ./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
  --ckpt ../weights/checkpoint_epoch_36_nus_mamba.pth \
  --batch_size 2 \
  --workers 4 \
  --set DATA_CONFIG.VERSION v1.0-mini