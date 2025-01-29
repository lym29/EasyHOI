import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

sys.path.append("./third_party/")
sys.path.append("./third_party/LISA")

from LISA.model.LISA import LISAForCausalLM
from LISA.model.llava import conversation as conversation_lib
from LISA.model.llava.mm_utils import tokenizer_image_token
from LISA.model.segment_anything.utils.transforms import ResizeLongestSide
from LISA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                        DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from tqdm import tqdm

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1-explanatory")
    parser.add_argument("--seg_hand", action="store_true", default=False)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, required=True, help="Provide the path to the data directory. The directory must contain a folder named 'images'.")
    
    
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def clear_mask(mask, min_area = 20):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    filtered_mask = np.zeros_like(mask)

    cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
    return mask

def main(args):
    args = parse_args(args)
    
    data_dir = os.path.abspath(args.data_dir)
    if args.seg_hand:
        vis_save_path = os.path.join(data_dir, "obj_recon/hand_mask")
    else:
        vis_save_path = os.path.join(data_dir, "obj_recon/obj_mask")
    
    os.makedirs(vis_save_path, exist_ok=True)
    
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
        
    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    
    
    image_dir = os.path.join(data_dir, "images")
    lisa_output_dir = os.path.join(data_dir, "LISA_output")
    os.makedirs(lisa_output_dir, exist_ok=True)

    for file in tqdm(os.listdir(image_dir)):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        if args.seg_hand:
            prompt = "Please segment all hands in this image."
        else:
            prompt = "What is being held by the hand? Please output a segmentation mask."
        
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = os.path.join(image_dir, file)
        if os.path.isdir(image_path):
            import shutil
            shutil.rmtree(image_path)
            print("Directory removed. ", image_path)
            continue
        
        image_name = file.split(".")[0]
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue
        save_path = os.path.join(vis_save_path, f"{image_name}.png")
        
        seg_type = "hand" if args.seg_hand else "obj"
        lisaout_path = os.path.join(lisa_output_dir, f"{image_name}_masked_{seg_type}_0.jpg")
        if args.skip and os.path.exists(lisaout_path):
            continue


        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0
            
            if args.seg_hand:
                save_path = os.path.join(vis_save_path, f"{image_name}.png")
                mask_image = (pred_mask == False) * 255 # hand part get black, others get white
                cv2.imwrite(save_path, mask_image)
                print("{} has been saved.".format(save_path))
                
                save_path = "{}/{}_masked_hand_{}.jpg".format(
                    lisa_output_dir, image_path.split("/")[-1].split(".")[0], i
                )
                color = np.array([255, 0, 0]) # red
            else:
                pred_mask = clear_mask(np.uint8(pred_mask) * 255)
                save_path = os.path.join(vis_save_path, f"{image_name}.png")
                cv2.imwrite(save_path, pred_mask)
                print("{} has been saved.".format(save_path))
                
                pred_mask = pred_mask>0

                save_path = "{}/{}_masked_obj_{}.jpg".format(
                    lisa_output_dir, image_path.split("/")[-1].split(".")[0], i
                )
                color = np.array([0, 255, 0]) # green
                
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * color * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
