import json
import os
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import zipfile

# === 설정 ===
base_dir = "data/imagereward"  # JSON과 이미지가 있는 루트
hf_site = "https://hf-mirror.com"
metadata_path = "ImageRewardDB/raw/metadata-large.parquet"  # part_id 정보 포함된 parquet


def check_corrupt_images(json_path, output_json_path):
    """Step 1: 손상된 이미지 탐지 및 기록"""

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            return json.load(f)

    corrupt_images = {}

    # 평가용 JSON 로드
    with open(json_path, 'r') as f:
        data = json.load(f)

    # part_id가 들어 있는 DiffusionDB 메타데이터 로드
    df = pd.read_parquet(metadata_path)

    for item in tqdm(data, desc="check corrupted images"):
        for img_path in item['image_path']:
            full_path = os.path.join(base_dir, img_path)
            try:
                img = Image.open(full_path)
                img.verify()
            except (IOError, SyntaxError, FileNotFoundError):
                filename = Path(full_path).name
                match = df[df['image_name'] == filename]
                if not match.empty:
                    part_id = int(match['part_id'].iloc[0])
                    if part_id not in corrupt_images:
                        corrupt_images[part_id] = {
                            'images': [],
                            'fixed': False
                        }
                    corrupt_images[part_id]['images'].append({
                        'full_path': full_path,
                        'filename': filename
                    })
                else:
                    print(f"⚠️ Warning: {filename} not found in metadata-large.parquet")

    # 결과 저장
    with open(output_json_path, 'w') as f:
        json.dump(corrupt_images, f, indent=2)

    return corrupt_images


def fix_corrupt_images(corrupt_images_dict, output_json_path):
    """Step 2: 손상된 이미지를 zip 파일에서 추출하여 복구"""
    os.makedirs('./tmp', exist_ok=True)

    # part_id(str), info(dict)
    for part_id_str, info in tqdm(corrupt_images_dict.items(), desc="fix corrupted images"):
        if info['fixed']:
            continue

        # zip 다운로드용으로만 int 변환
        part_id = int(part_id_str)

        zip_url = (
            f'{hf_site}/datasets/poloclub/diffusiondb/resolve/main/'
            f'diffusiondb-large-part-1/part-{part_id:06}.zip'
        )
        temp_zip = f'./tmp/part-{part_id:06}.zip'

        try:
            # zip 파일 다운로드
            cmd = f'aria2c -c "{zip_url}" -d ./tmp -o "part-{part_id:06}.zip"'
            if os.system(cmd) != 0:
                raise Exception("aria2c download failed")

            # zip 파일 열기
            with zipfile.ZipFile(temp_zip) as zip_ref:
                for img_info in info['images']:
                    full_path = img_info['full_path']
                    filename = img_info['filename']

                    # 디렉토리 보장
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    # zip 안의 파일로 복구
                    with zip_ref.open(filename) as source:
                        with open(full_path, 'wb') as target:
                            target.write(source.read())

            # zip 삭제
            os.remove(temp_zip)

            # 🔥 여기서 문자열 키 그대로 사용
            corrupt_images_dict[part_id_str]['fixed'] = True

            # 중간 저장
            with open(output_json_path, 'w') as f:
                json.dump(corrupt_images_dict, f, indent=2)

        except Exception as e:
            print(f"❌ Error for part {part_id}: {str(e)}")
            continue

    return corrupt_images_dict



# === 메인 실행 ===
if __name__ == "__main__":
    json_path = os.path.join(base_dir, "train.json")
    output_json_path = os.path.join(base_dir, "corrupt_images.json")

    # Step 1: 손상 이미지 탐지
    corrupt_images = check_corrupt_images(json_path, output_json_path)

    # Step 2: 손상 이미지 복구
    fixed_results = fix_corrupt_images(corrupt_images, output_json_path)

    # 요약 출력
    print("\n🧾 복구 요약:")
    for part_id, info in fixed_results.items():
        if info['fixed']:
            print(f"✅ Part-{part_id}: fixed")
        else:
            print(f"❌ Part-{part_id}: {len(info['images'])} images not fixed")
