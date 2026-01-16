import argparse
import json
import os
import sys
import traceback
from typing import List

import torch
from diffusers import FluxPipeline
from tqdm import tqdm

def list_task_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def main():
    parser = argparse.ArgumentParser(
        description="flux inference"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/path/to/input",
        help="input path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/output",
        help="output path",
    )
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="classifier-free guidance")
    parser.add_argument("--height", type=int, default=256, help="image height")
    parser.add_argument("--width", type=int, default=256, help="image width")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="denoising steps")
    parser.add_argument("--use_cpu_offload", type=bool, default=False, help="Whether to use CPU offload.")

    args = parser.parse_args()

    shot_input_path = os.path.join(args.input_dir, f"shot_{args.shot}")
    shot_output_path = os.path.join(args.output_dir, f"shot_{args.shot}_flux")
    os.makedirs(shot_output_path, exist_ok=True)

    if not os.path.isdir(shot_input_path):
        print(f"[ERROR] input path not exists：{shot_input_path}", file=sys.stderr)
        sys.exit(1)

    pipe = FluxPipeline.from_pretrained(
        "blackforest/FLUX1.dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map="balanced",
    )

    pipe.set_progress_bar_config(disable=True)
    if args.use_cpu_offload:
        pipe.enable_model_cpu_offload()

    task_names = list_task_dirs(shot_input_path)

    for task_name in task_names:
        in_task_dir = os.path.join(shot_input_path, task_name)
        out_task_dir = os.path.join(shot_output_path, task_name)
        os.makedirs(out_task_dir, exist_ok=True)

        file_names = [f for f in os.listdir(in_task_dir) if os.path.isfile(os.path.join(in_task_dir, f))]
        to_process = []
        for fname in file_names:
            base, _ = os.path.splitext(os.path.basename(fname))
            target_img = os.path.join(out_task_dir, f"{base}.jpg")
            if os.path.exists(target_img):
                continue
            to_process.append(fname)

        pbar = tqdm(to_process, desc=f"Generating {task_name}", unit="file")
        for file_name in pbar:
            file_path = os.path.join(in_task_dir, file_name)
            base, _ = os.path.splitext(os.path.basename(file_name))
            image_path = os.path.join(out_task_dir, f"{base}.jpg")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_name.lower().endswith(".json"):
                        data = json.load(f)
                        description = data.get("description", "")
                        if isinstance(description, list):
                            description = description[0] if description else ""

                if not description:
                    raise ValueError("Empty prompt/description")

                with torch.inference_mode():
                    out = pipe(
                        prompt=description,
                        num_images_per_prompt=1,
                        guidance_scale=args.guidance_scale,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                    )
                    image = out.images[0]

                image.save(image_path)

                pbar.set_postfix_str("ok")

            except Exception as e:
                err_msg = f"[FAIL] {task_name}/{file_name}: {repr(e)}"
                print(err_msg, file=sys.stderr)
                traceback.print_exc()

    print("DONE.")


if __name__ == "__main__":
    main()
