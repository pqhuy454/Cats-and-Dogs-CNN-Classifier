from pathlib import Path
import cv2
import shutil
import hashlib

# Xác định thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
SOURCE_DIR = PROJECT_DIR / "data" / "PetImages"
OUTPUT_DIR = PROJECT_DIR / "data" / "PetImages_cleaned"

# Tên thư mục nguồn và tên lớp đầu ra
CLASSES = {
    "Cat": "cat",
    "Dog": "dog",
}


# Tính mã băm MD5 của ảnh
def compute_hash(image_path):
    hasher = hashlib.md5()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# Kiểm tra ảnh có đọc được hay không
def is_valid_image(image_path):
    image = cv2.imread(str(image_path))
    return image is not None


# Tạo các thư mục đầu ra
def prepare_output_folders():
    for output_class in CLASSES.values():
        output_path = OUTPUT_DIR / output_class
        output_path.mkdir(parents=True, exist_ok=True)


# Làm sạch ảnh của một lớp
def clean_class(source_class, output_class):
    src_dir = SOURCE_DIR / source_class
    dst_dir = OUTPUT_DIR / output_class

    if not src_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {src_dir}"
        )

    seen_hashes = set()
    removed = 0
    kept = 0

    for image_path in src_dir.iterdir():
        if not image_path.is_file():
            continue

        # Bỏ qua ảnh không đọc được
        if not is_valid_image(image_path):
            removed += 1
            continue

        # Tính mã băm và loại ảnh trùng trong cùng lớp
        image_hash = compute_hash(image_path)

        if image_hash in seen_hashes:
            removed += 1
            continue

        seen_hashes.add(image_hash)

        # Sao chép ảnh hợp lệ sang thư mục đầu ra
        shutil.copy2(
            image_path,
            dst_dir / image_path.name
        )
        kept += 1

    print(
        f"{source_class}: giữ lại {kept} ảnh, "
        f"bỏ qua {removed} ảnh lỗi/trùng."
    )


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {SOURCE_DIR}"
        )

    # Xóa dữ liệu đầu ra cũ để tạo lại từ đầu
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    prepare_output_folders()

    for source_class, output_class in CLASSES.items():
        clean_class(source_class, output_class)

    print(f"\nDữ liệu đã được làm sạch tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()