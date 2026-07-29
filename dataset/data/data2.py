import random
from pathlib import Path

# ==========================
# 配置
# ==========================

IMG_DIRS = [
    "img1",
    "img2",
    "img3"
]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42

# ==========================
# 重新编号
# ==========================

def renumber_folder(folder):

    folder = Path(folder)

    if not folder.exists():
        print(f"{folder} 不存在")
        return 0

    files = sorted(folder.glob("*.jpg"))

    if len(files) == 0:
        print(f"{folder} 无jpg文件")
        return 0

    # 第一阶段：临时改名
    temp_files = []

    for i, file in enumerate(files):

        temp_name = folder / f"temp_{i:04d}.jpg"

        file.rename(temp_name)

        temp_files.append(temp_name)

    # 第二阶段：正式编号
    count = 0

    for idx, temp_file in enumerate(temp_files, start=1):

        new_name = folder / f"{idx:04d}.jpg"

        temp_file.rename(new_name)

        count += 1

    print(
        f"{folder.name}: "
        f"重新编号完成，共 {count} 张"
    )

    return count

# ==========================
# 生成all_images.txt
# ==========================

def generate_all_images():

    all_images = []

    for folder in IMG_DIRS:

        folder = Path(folder)

        if not folder.exists():
            continue

        files = sorted(folder.glob("*.jpg"))

        for file in files:

            all_images.append(
                file.as_posix()
            )

    with open(
        "all_images.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for item in all_images:
            f.write(item + "\n")

    print(
        f"all_images.txt 已生成 "
        f"({len(all_images)} 张)"
    )

    return all_images

# ==========================
# 划分数据集
# ==========================

def split_dataset(all_images):

    random.seed(RANDOM_SEED)

    random.shuffle(all_images)

    total = len(all_images)

    train_num = int(total * TRAIN_RATIO)
    val_num = int(total * VAL_RATIO)

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
    ) as f:

        f.write("\n".join(train_set))

    with open(
        "val.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(val_set))

    with open(
        "test.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(test_set))

    print()
    print("数据集划分完成")
    print(f"Train : {len(train_set)}")
    print(f"Val   : {len(val_set)}")
    print(f"Test  : {len(test_set)}")

# ==========================
# 主程序
# ==========================

if __name__ == "__main__":

    print("开始重新编号...\n")

    total_count = 0

    for folder in IMG_DIRS:

        total_count += renumber_folder(folder)

    print()
    print(f"总图片数: {total_count}")

    print("\n生成txt文件...")

    all_images = generate_all_images()

    split_dataset(all_images)

    print("\n全部完成")