'''
Calendula597
'''
import argparse
import json
import os
from time import time

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, InternVLForConditionalGeneration, AutoTokenizer

from cobsat.load_dataset import load_dataset, get_prompt


# ------------------------- 主流程 -------------------------
def main():
    parser = argparse.ArgumentParser()
    # 输出设置
    parser.add_argument("--out_dir", default="/home/sdbdata/Lizhenpeng/experiment/glm4vbase_2shot",
                        help="output directory")
    parser.add_argument("--model_name", default='qwen2vl', type=str, choices=['qwen2vl', 'qwen2.5vl', 'internvl3'],
                        help="model name")
    parser.add_argument("--task_id", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int, nargs='+', help="任务ID")
    parser.add_argument("--shot", type=int, default=2, help="Few-shot数量")
    parser.add_argument("--prompt_type", default="default", type=str)
    parser.add_argument("--data_mode", default="default", type=str)

    args = parser.parse_args()
    model_name = args.model
    shot_dir = os.path.join(args.out_dir, f"shot_{args.shot}")
    os.makedirs(shot_dir, exist_ok=True)

    instruction = "I give you several words and pictures. First, please analyse what the next picture is. Then give me a " \
                  "detailed diffusion prompt to describe the next picture. Please only provide me the detailed prompt " \
                  "and start the answer with 'Create an image'. Note: " \
                  "Focus mainly on understanding and following the meaning of the " \
                  "final text when creating your description.\n\n"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", local_files_only=True,
                                              trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", local_files_only=True,
                                              trust_remote_code=True)

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
    """
        📂 文件夹结构说明
        xx_shot/                 # 实验目录
        │
        └── task_x/                # 特征文件存放
            ├── *.json 
            └──  *.pth
    """

    for task_id in args.task_id:
        task_dir = os.path.join(shot_dir, f"task_{task_id}")
        os.makedirs(shot_dir, exist_ok=True)

        # 初始化数据集
        data_loader = load_dataset(
            args.shot,
            args.prompt_type,
            task_id,
            data_mode=args.data_mode,
            ft_mode="all",
        )

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

            placeholders.append({"type": "text", "text": query["text_inputs"][-1]})

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
            images, _ = process_vision_info(messages)
            inputs = processor(text=text, images=images, return_tensors="pt").to(lvlm.device)

            start_time = time()

            outputs = lvlm.generate(**inputs, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids.cpu(), outputs['sequences'])
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            end_time = time()

            output_dict = {"instruction": instruction, "time": end_time - start_time, "description": output_text,
                           "image_inputs": image_inputs,
                           "text_inputs": text_inputs}

            json_path = os.path.join(task_dir, f"{input_dict['save_path']}.json")

            with open(json_path, 'w', encoding="utf-8") as f:
                json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
