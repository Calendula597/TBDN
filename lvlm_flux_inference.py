import argparse
import json
import os
from time import time

import torch
from cobsat.load_dataset import load_dataset, get_prompt
from diffusers import FluxPipeline
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, \
    InternVLForConditionalGeneration, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    # output setting
    parser.add_argument("--out_dir", default="/path/to/output", help="output path")
    # LVLM setting
    parser.add_argument("--model", default="qwen2vl", choices=['qwen2vl', 'qwen2_5vl', 'internvl3'], help="LVLM model")
    parser.add_argument("--use_qcd", default=False, type=bool, help="whether to use qcd")
    parser.add_argument("--alpha", default=0.5, type=float, help="alpha value")

    # CoBSAT setting
    parser.add_argument("--task_id", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int, nargs='+', help="任务ID")
    parser.add_argument("--shot", type=int, default=2, help="number of shot")
    parser.add_argument("--prompt_type", default="default", type=str)
    parser.add_argument("--data_mode", default="default", type=str)

    # flux setting
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="guidance scale")
    parser.add_argument("--height", type=int, default=256, help="image height")
    parser.add_argument("--width", type=int, default=256, help="image width")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="number of inference steps")
    parser.add_argument("--use_cpu_offload", type=bool, default=True, help="Whether to use CPU offload.")

    args = parser.parse_args()

    shot_dir = os.path.join(args.out_dir, f"shot_{args.shot}")
    shot_flux_dir = os.path.join(args.out_dir, f"shot_{args.shot}_flux")

    os.makedirs(shot_dir, exist_ok=True)
    os.makedirs(shot_flux_dir, exist_ok=True)

    instruction = "I give you several words and pictures. First, please analyse what the next picture is. Then give me a " \
                  "detailed diffusion prompt to describe the next picture. Please only provide me the detailed prompt " \
                  "and start the answer with 'Create an image'.\n\n"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

    if args.model == "qwen2vl":
        lvlm = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    elif args.model == "qwen2_5vl":
        lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    elif args.model == "internvl3":
        lvlm = InternVLForConditionalGeneration.from_pretrained(
            "InternLM/internlm-3b-vision", torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    else:
        raise ValueError("Invalid model name")

    lvlm.eval()

    flux = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(lvlm.device)

    if args.use_cpu_offload:
        flux.enable_model_cpu_offload()

    """
        📂 
        path_to_output/                 # 实验目录
        │
        │└── shot_x/                # 特征文件存放
            ├── *.json 
            └──  *.pth
    """

    for task_id in args.task_id:
        task_dir = os.path.join(shot_dir, f"task_{task_id}")
        task_flux_dir = os.path.join(shot_flux_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(task_flux_dir, exist_ok=True)

        # load CoBSAT dataset
        data_loader = load_dataset(
            args.shot,
            args.prompt_type,
            task_id,
            data_mode=args.data_mode,
            ft_mode="all",
        )

        # inference
        for input_dict in tqdm(data_loader, desc=f"Extract CoBSAT task_id: {task_id} shot: {args.shot} feats and text",
                               total=len(data_loader)):
            text_inputs, image_inputs = input_dict["text_inputs"], input_dict["image_inputs"]
            query = get_prompt(
                text_inputs,
                image_inputs,
                args.prompt_type,
                task_id,
                "qwen",
                "text",
            )

            placeholders = []
            for i in range(len(query["image_inputs"])):
                placeholders.append({"type": "text", "text": query["text_inputs"][i]})
                placeholders.append(
                    {
                        "type": "image",
                        "image": query["image_inputs"][i],
                    }
                )

            full_messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role":
                    "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                    *placeholders,
                    {"type": "text", "text": query["text_inputs"][-1]}
                ],
            }]

            full_text = processor.apply_chat_template(full_messages,
                                                      tokenize=False,
                                                      add_generation_prompt=True,
                                                      add_vision_id=True
                                                      )
            images, _ = process_vision_info(full_messages)

            full_inputs = processor(text=full_text, images=images, return_tensors="pt").to(lvlm.device)

            gen_kwargs = dict(
                max_new_tokens=128,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=False,
                use_cache=True,
                output_attentions=False,
                output_scores=False,
            )
            if args.use_qcd:
                messages = [{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role":
                        "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction
                        },
                        *placeholders,
                    ],
                }]
                text = processor.apply_chat_template(messages,
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     add_vision_id=True
                                                     )
                inputs = processor(text=text, images=images, return_tensors="pt").to(lvlm.device)
                gen_kwargs['input_ids_cd'] = inputs
                gen_kwargs['cd_alpha'] = args.alpha

            start_time = time()
            with torch.inference_mode():
                lvlm_output = lvlm.generate(**full_inputs, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(full_inputs.input_ids, lvlm_output['sequences'])
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            with torch.inference_mode():
                flux_output = flux(
                    prompt=output_text,
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                )

            image = flux_output.images[0]

            end_time = time()

            json_path = os.path.join(task_dir, f"{input_dict['save_path']}.json")
            image_path = os.path.join(task_flux_dir, f"{input_dict['save_path']}.jpg")
            image.save(image_path)

            output_dict = {"instruction": instruction, "time": end_time - start_time, "description": output_text,
                           "image_inputs": image_inputs,
                           "text_inputs": text_inputs, "image": image_path}

            with open(json_path, 'w', encoding="utf-8") as f:
                json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
