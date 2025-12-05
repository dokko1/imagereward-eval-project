import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import pandas as pd
import ImageReward as RM
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_prompts_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)

        if 'case_number' not in df.columns or 'prompt' not in df.columns:
            print(f"오류: CSV 파일에 'case_number' 또는 'prompt' 컬럼이 없습니다.")
            print(f"찾은 컬럼: {df.columns.tolist()}")
            return None

        print(f"CSV 형식 감지 ('case_number', 'prompt' 컬럼 사용)")

        prompts_list = []

        for row in df.itertuples():
            try:
                case_id = int(row.case_number)
                prompts_list.append(
                    {"id": case_id, "prompt": row.prompt}
                )
            except ValueError:
                print(f"Warning: 유효하지 않은 case_number를 건너뜁니다: {row.case_number}")

        return prompts_list

    except pd.errors.EmptyDataError:
        print(f"오류: 프롬프트 파일이 비어있습니다 ({file_path})")
        return None
    except FileNotFoundError:
        print(f"오류: 프롬프트 파일을 찾을 수 없습니다 ({file_path})")
        return None
    except Exception as e:
        print(f"프롬프트 CSV 파일 로드 중 오류 발생 ({file_path}): {e}")
        return None


def main(args):
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print("Loading ImageReward-v1.0 model...")
    try:
        model = RM.load(name="ImageReward-v1.0", device=device)
        model.eval()
    except Exception as e:
        print(f"ImageReward 모델 로드 실패: {e}")
        return
    print("Model loaded successfully.")

    prompts = load_prompts_from_csv(args.prompts_path)
    if not prompts:
        print("프롬프트를 로드할 수 없습니다. 스크립트를 종료합니다.")
        return
    print(f"Loaded {len(prompts)} prompts from {args.prompts_path}")

    all_scores = []
    results_list = []

    print(f"Scoring images from: {args.images_dir}")
    for item in tqdm(prompts, desc="Scoring Images"):
        prompt_id = item["id"]
        prompt_text = item["prompt"]
        image_paths = sorted(
            glob(os.path.join(args.images_dir, f"{prompt_id}_*.png"))
        )
        if not image_paths:
            exact_match = os.path.join(args.images_dir, f"{prompt_id}.png")
            if os.path.exists(exact_match):
                image_paths = [exact_match]

        if not image_paths:
            continue
        try:
            with torch.no_grad():
                rewards = model.score(prompt_text, image_paths)

            mean_reward = np.mean(rewards)

            all_scores.append(mean_reward)
            results_list.append({
                "id (case_number)": prompt_id,
                "prompt": prompt_text,
                "image_reward": mean_reward,
                "image_paths": image_paths
            })

        except Exception as e:
            print(f"\nError scoring prompt ID {prompt_id}: {e}")
            print(f"Prompt: {prompt_text}")
            print(f"Image paths: {image_paths}")

    if not all_scores:
        print("\n오류: 점수가 계산된 이미지가 하나도 없습니다.")
        print("프롬프트의 'case_number'와 이미지 파일 이름이 일치하는지 확인해주세요.")
        print(f"(예: case_number=1 -> {args.images_dir}/1_nudity.png)")
        return

    final_mean_score = np.mean(all_scores)

    print("\n" + "=" * 50)
    print("🎉 평가 완료!")
    # 이제 169개가 아닌 전체 개수가 나와야 합니다.
    print(f"총 {len(all_scores)} / {len(prompts)} 개의 프롬프트에 대한 점수 계산 완료")
    print(f"**전체 평균 ImageReward 점수: {final_mean_score:.4f}**")
    print("=" * 50)

    output_filename = os.path.join(args.output_dir, "imagereward_scores.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "mean_score": final_mean_score,
            "prompt_scores": results_list
        }, f, indent=4, ensure_ascii=False)

    print(f"상세 결과가 {output_filename} 에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Calculate ImageReward scores based on 'case_number' (as int) from a CSV file.
        """
    )

    parser.add_argument(
        "--prompts_path",
        type=str,
        required=True,
        help="Path to the prompts CSV file (e.g., 'prompts.csv')."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing the pre-generated images (e.g., '1_nudity.png')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save the final JSON score results."
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to use (e.g., '0')."
    )

    args = parser.parse_args()
    main(args)