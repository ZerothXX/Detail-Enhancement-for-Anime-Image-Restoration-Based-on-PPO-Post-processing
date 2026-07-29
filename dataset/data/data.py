import random
from pathlib import Path

from PIL import Image, ImageSequence

# =====================================
# 配置
# =====================================

INPUT_OUTPUT_MAP = {
    "part1": "img1",
    "part2": "img2",
    "part3": "img3",
}

MIN_SIZE = 256
MAX_RATIO = 2.0

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42

# =====================================
# 判断是否保留
# =====================================

def is_valid_image(width, height):

    if min(width, height) < MIN_SIZE:
        return False

    ratio = max(width, height) / min(width, height)

    if ratio > MAX_RATIO:
        return False

    return True


# =====================================
# 读取图片
# =====================================

def load_image(path):

    ext = path.suffix.lower()

    try:

        if ext == ".gif":

            gif = Image.open(path)

            try:
                frame = next(ImageSequence.Iterator(gif))
            except:
                frame = gif

            img = frame.convert("RGB")

        else:

            img = Image.open(path).convert("RGB")

        return img

    except:
        return None


# =====================================
# 处理一个目录
# =====================================

def process_folder(src_dir, dst_dir):

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif"
    }

    files = []

    for f in src_dir.iterdir():
        if f.suffix.lower() in exts:
            files.append(f)

    files.sort()

    save_count = 1

    total = 0
    removed = 0
    saved = 0

    for file_path in files:

        total += 1

        img = load_image(file_path)

        if img is None:
            removed += 1
            continue

        w, h = img.size

        if not is_valid_image(w, h):
            removed += 1
            continue

        new_name = f"{save_count:05d}.jpg"

        save_path = dst_dir / new_name

        img.save(
            save_path,
            "JPEG",
            quality=95
        )

        save_count += 1
        saved += 1

        if saved % 500 == 0:
            print(
                f"{dst_dir.name}: "
                f"已保存 {saved} 张"
            )

    print("\n" + "=" * 60)
    print(f"{src_dir.name} 处理完成")
    print(f"总数: {total}")
    print(f"保留: {saved}")
    print(f"删除: {removed}")
    print("=" * 60)

    return saved


# =====================================
# 生成 all_images.txt
# =====================================

def generate_all_images():

    all_images = []

    for out_dir in INPUT_OUTPUT_MAP.values():

        out_dir = Path(out_dir)

        if not out_dir.exists():
            continue

        files = sorted(out_dir.glob("*.jpg"))

        for f in files:

            rel_path = f.as_posix()

            all_images.append(rel_path)

    with open(
        "all_images.txt",
        "w",
        encoding="utf-8"
    ) as fp:

        for item in all_images:
            fp.write(item + "\n")

    print(
        f"\nall_images.txt 已生成 "
        f"({len(all_images)} 张)"
    )

    return all_images


# =====================================
# 划分 train val test
# =====================================

def split_dataset(all_images):

    random.seed(RANDOM_SEED)

    random.shuffle(all_images)

    n = len(all_images)

    train_num = int(n * TRAIN_RATIO)
    val_num = int(n * VAL_RATIO)

    train_set = all_images[:train_num]

    val_set = all_images[
        train_num:
        train_num + val_num
    ]

    test_set = all_images[
        train_num + val_num:
    ]

    with open(
        "train.txt",
        "w",
        encoding="utf-8"
    ) as fp:

        fp.write("\n".join(train_set))

    with open(
        "val.txt",
        "w",
        encoding="utf-8"
    ) as fp:

        fp.write("\n".join(val_set))

    with open(
        "test.txt",
        "w",
        encoding="utf-8"
    ) as fp:

        fp.write("\n".join(test_set))

    print("\n数据集划分完成")
    print(f"Train : {len(train_set)}")
    print(f"Val   : {len(val_set)}")
    print(f"Test  : {len(test_set)}")


# =====================================
# 主程序
# =====================================

if __name__ == "__main__":

    print("开始处理图片...\n")

    total_saved = 0

    for src, dst in INPUT_OUTPUT_MAP.items():

        if not Path(src).exists():

            print(f"未找到目录: {src}")
            continue

        total_saved += process_folder(
            src,
            dst
        )

    print(
        f"\n全部处理完成，"
        f"保留图片总数: {total_saved}"
    )

    all_images = generate_all_images()

    split_dataset(all_images)

    print("\n全部完成")