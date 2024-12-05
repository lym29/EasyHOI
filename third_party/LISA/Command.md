
conda activate llmseg
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1-explanatory' --precision='fp16' --load_in_8bit


CUDA_VISIBLE_DEVICES=0 python chat_arctic.py --version='xinlai/LISA-13B-llama2-v1-explanatory' --precision='fp16' --seg_hand --load_in_8bit


CUDA_VISIBLE_DEVICES=0 python chat_arctic.py --version='xinlai/LISA-13B-llama2-v1-explanatory' --precision='fp16' --load_in_8bit