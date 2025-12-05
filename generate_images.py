import argparse
import json
import os
import torch
import pandas as pd
from tqdm import tqdm
import warnings
from diffusers import DiffusionPipeline
from diffusers import VersatileDiffusionTextToImagePipeline

def load_prompts(file_path, id_column, prompt_column, seed_column):
    print(f"Loading prompts from: {file_path}")
    print(f"Using columns -> ID: '{id_column}', Prompt: '{prompt_column}', Seed: '{seed_column}'")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    required = [id_column, prompt_column, seed_column]
    for col in required:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'. Found: {df.columns.tolist()}")
            return None

    prompts_list = []
    for row in df.itertuples():
        try:
            item = {
                "id": int(getattr(row, id_column)),
                "prompt": str(getattr(row, prompt_column)),
                "seed": int(getattr(row, seed_column)),
            }
            prompts_list.append(item)
        except:
            continue

    print(f"Loaded {len(prompts_list)} prompts.")
    return prompts_list

def disable_safety(pipe):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "feature_extractor"):
        pipe.feature_extractor = None
    return pipe


def main(args):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available.")

    warnings.filterwarnings("ignore", category=UserWarning)

    # ─ Load prompts ──────────────────────
    prompts = load_prompts(
        args.prompts_path,
        args.id_column,
        args.prompt_column,
        args.seed_column
    )
    if not prompts:
        return

    # ─ Load model list ─────────────────────
    try:
        model_map = json.loads(args.model_map)
        print(f"Models to process: {list(model_map.keys())}")
    except json.JSONDecodeError:
        print("Error: --model_map must be valid JSON.")
        return

    # Guidance scale list
    guidance_list = [1.5,2,3,4,5]
    #guidance_list = [3, 4, 5]

    for model_name, model_id in model_map.items():
        print("\n" + "=" * 70)
        print(f"Processing Model: {model_name} ({model_id})")
        print("=" * 70)

        # ─ Load model ───────────────────
        try:
            if model_id == "shi-labs/versatile-diffusion":
                print("Loading Versatile Diffusion...")
                pipe = VersatileDiffusionTextToImagePipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
                pipe.remove_unused_weights()
            else:
                print(f"Loading DiffusionPipeline: {model_id}")
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    low_cpu_mem_usage=False,
                    device_map=None
                )

            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            pipe = disable_safety(pipe)

        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            continue

        # ─ Output base folder ───────────────────
        base_folder = f"{model_name}_coco_small_with_seed"
        model_output_dir = os.path.join(args.output_dir, base_folder)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"Saving images under: {model_output_dir}")

        for gs in guidance_list:
            print(f"\n→ Starting guidance scale {gs}")

            gs_folder_name = f"gs{gs}"
            gs_output_dir = os.path.join(model_output_dir, gs_folder_name)
            os.makedirs(gs_output_dir, exist_ok=True)

            record_items = []

            for item in tqdm(prompts, desc=f"GS {gs} ({model_name})"):
                prompt_id = item["id"]
                prompt_text = item["prompt"]
                prompt_seed = item["seed"]

                output_filename = f"{prompt_id}.png"
                output_filepath = os.path.join(gs_output_dir, output_filename)

                if os.path.exists(output_filepath) and not args.force_regenerate:
                    record_items.append({
                        "id": prompt_id,
                        "seed": prompt_seed,
                        "filename": f"{gs_folder_name}/{output_filename}"
                    })
                    continue

                try:
                    generator = torch.Generator(device=device).manual_seed(prompt_seed)

                    with torch.no_grad():
                        result = pipe(
                            prompt=prompt_text,
                            num_images_per_prompt=1,
                            height=256,
                            width=256,
                            guidance_scale=gs,
                            generator=generator
                        ).images

                    result[0].save(output_filepath)
                    del result

                    record_items.append({
                        "id": prompt_id,
                        "seed": prompt_seed,
                        "filename": f"{gs_folder_name}/{output_filename}"
                    })

                except Exception as e:
                    print(f"\nError prompt_id {prompt_id}: {e}")

            json_path = os.path.join(
                model_output_dir,
                f"{model_name}_coco_small_with_seed_gs{gs}.json"
            )
            with open(json_path, "w") as f:
                json.dump({
                    "model": model_name,
                    "guidance_scale": gs,
                    "items": record_items
                }, f, indent=4)

            print(f"Saved JSON: {json_path}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Unloaded {model_name}.")

    print("\n" + "=" * 70)
    print("🎉 All tasks complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--id_column", type=str, required=True)
    parser.add_argument("--prompt_column", type=str, required=True)
    parser.add_argument("--seed_column", type=str, required=True)

    parser.add_argument("--model_map", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="generated_images")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--force_regenerate", action="store_true")

    args = parser.parse_args()
    main(args)
