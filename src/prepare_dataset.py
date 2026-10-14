from pathlib import Path
import shutil
import random
import cv2

# Xác định thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
SOURCE_DIR = PROJECT_DIR / "data" / "PetImages_cleaned"
OUTPUT_DIR = PROJECT_DIR / "data" / "dataset"

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Kích thước ảnh đầu vào
IMG_SIZE = (150, 150)

# Tên các lớp
CLASSES = ["cat", "dog"]

# Seed để tái lập kết quả chia dữ liệu
SEED = 42

random.seed(SEED)


def prepare_folders():
    """Tạo thư mục train, validation và test."""
    for split in ["train", "validation", "test"]:
        for cls in CLASSES:
            path = OUTPUT_DIR / split / cls
            path.mkdir(parents=True, exist_ok=True)


def resize_and_save(src_path, dst_path):
    """Đọc, resize và lưu ảnh."""
    img = cv2.imread(str(src_path))

    if img is None:
        return False

    resized_img = cv2.resize(img, IMG_SIZE)

    success = cv2.imwrite(str(dst_path), resized_img)
    return success


def split_and_copy():
    """Chia ảnh theo tỷ lệ 70/15/15 và lưu vào các tập."""
    for cls in CLASSES:
        src_dir = SOURCE_DIR / cls

        if not src_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu: {src_dir}"
            )

        # Chỉ lấy các file ảnh
        images = [
            path for path in src_dir.iterdir()
            if path.is_file()
        ]

        random.shuffle(images)

        total_images = len(images)

        # Chia dữ liệu
        train_size = int(total_images * TRAIN_RATIO)
        val_size = int(total_images * VAL_RATIO)

        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        splits = {
            "train": train_images,
            "validation": val_images,
            "test": test_images,
        }

        # Resize và lưu ảnh vào từng tập
        for split_name, split_images in splits.items():
            for image_path in split_images:
                dst_path = (
                    OUTPUT_DIR / split_name / cls / image_path.name
                )

                if not resize_and_save(image_path, dst_path):
                    print(f"Bỏ qua ảnh không đọc/lưu được: {image_path}")

        print(
            f"{cls}: tổng {total_images} ảnh | "
            f"train {len(train_images)} | "
            f"validation {len(val_images)} | "
            f"test {len(test_images)}"
        )


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu nguồn: {SOURCE_DIR}\n"
            "Hãy chạy clean_dataset.py trước."
        )

    # Kiểm tra tổng tỷ lệ
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Tổng tỷ lệ train/validation/test phải bằng 1.")

    # Xóa dữ liệu đầu ra cũ
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    prepare_folders()
    split_and_copy()

    print(f"\nĐã chia và chuẩn hóa dữ liệu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()