import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


#Cấu hình đường dẫn
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "model_CNN.keras"
TEST_DIR = PROJECT_DIR / "data" / "dataset" / "test"

RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "prediction_results.csv"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_FILE = RESULTS_DIR / "classification_report.csv"
FAILED_FILE = RESULTS_DIR / "failed_images.csv"

IMG_SIZE = (150, 150)
THRESHOLD = 0.5

#Nhãn lớp: cat = 0, dog = 1
CLASSES = {
    "cat": 0,
    "dog": 1
}

CLASS_NAMES = ["cat", "dog"]

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif"
}


#Kiểm tra đường dẫn
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}\n"
        "Hãy chạy training.py trước."
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy tập test: {TEST_DIR}\n"
        "Hãy chạy prepare_dataset.py trước."
    )

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


#Tải mô hình
model = load_model(str(MODEL_PATH))

print(f"Tải mô hình thành công: {MODEL_PATH}")


#Dự đoán ảnh
results = []
failed_images = []

for cls_name, true_label in CLASSES.items():

    cls_dir = TEST_DIR / cls_name

    if not cls_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục lớp: {cls_dir}"
        )

    image_paths = sorted(
        path for path in cls_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        print(f"Cảnh báo: Không có ảnh trong {cls_dir}")

    for img_path in image_paths:
        try:
            img = image.load_img(
                str(img_path),
                target_size=IMG_SIZE
            )

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            #Sigmoid trả về xác suất thuộc lớp dog
            prediction_score = float(
                model.predict(x, verbose=0)[0][0]
            )

            predicted_label = (
                1 if prediction_score >= THRESHOLD else 0
            )

            predicted_class = (
                "dog" if predicted_label == 1 else "cat"
            )

            is_correct = int(predicted_label == true_label)

            results.append({
                "filename": img_path.name,
                "filepath": str(
                    img_path.relative_to(PROJECT_DIR)
                ),
                "probability": prediction_score,
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "true_class": cls_name,
                "true_label": true_label,
                "correct": is_correct
            })

            print(
                f"{img_path.name}: "
                f"P(dog)={prediction_score:.4f} "
                f"-> {predicted_class} "
                f"({'đúng' if is_correct else 'sai'})"
            )

        except Exception as error:
            failed_images.append({
                "filename": img_path.name,
                "filepath": str(img_path),
                "error": str(error)
            })

            print(
                f"Lỗi khi xử lý {img_path.name}: {error}"
            )


#Kiểm tra kết quả
if not results:
    raise RuntimeError(
        "Không có ảnh nào được dự đoán thành công. "
        "Hãy kiểm tra thư mục test và định dạng ảnh."
    )

df = pd.DataFrame(results)

y_true = df["true_label"].astype(int)
y_pred = df["predicted_label"].astype(int)

#Chỉ số tổng thể
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    pos_label=1,
    zero_division=0
)

#Chỉ số macro: tính riêng từng lớp rồi lấy trung bình
precision_macro = precision_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

#Chỉ số weighted: trung bình có trọng số theo số mẫu mỗi lớp
precision_weighted = precision_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true, y_pred,
    average="weighted",
    zero_division=0
)


#Thống kê theo từng lớp
print("\n========== THỐNG KÊ THEO LỚP ==========")

for cls_name, true_label in CLASSES.items():

    class_df = df[df["true_label"] == true_label]

    total = len(class_df)
    correct = int(class_df["correct"].sum())
    wrong = total - correct

    class_accuracy = (
        correct / total if total > 0 else 0.0
    )

    class_precision = precision_score(
        y_true, y_pred,
        labels=[true_label],
        average="macro",
        zero_division=0
    )

    class_recall = recall_score(
        y_true, y_pred,
        labels=[true_label],
        average="macro",
        zero_division=0
    )

    class_f1 = f1_score(
        y_true, y_pred,
        labels=[true_label],
        average="macro",
        zero_division=0
    )

    print(f"\nLớp: {cls_name}")
    print(f"  Số ảnh:    {total}")
    print(f"  Dự đoán đúng: {correct}")
    print(f"  Dự đoán sai:  {wrong}")
    print(f"  Accuracy:  {class_accuracy:.4f}")
    print(f"  Precision: {class_precision:.4f}")
    print(f"  Recall:    {class_recall:.4f}")
    print(f"  F1-Score:  {class_f1:.4f}")


#In các chỉ số
print("\n========== CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH ==========")
print(f"Total samples:       {len(df)}")
print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision (dog):     {precision:.4f}")
print(f"Recall (dog):        {recall:.4f}")
print(f"F1-Score (dog):      {f1:.4f}")

print("\n--- Macro Average ---")
print(f"Precision Macro:     {precision_macro:.4f}")
print(f"Recall Macro:        {recall_macro:.4f}")
print(f"F1-Score Macro:      {f1_macro:.4f}")

print("\n--- Weighted Average ---")
print(f"Precision Weighted:  {precision_weighted:.4f}")
print(f"Recall Weighted:     {recall_weighted:.4f}")
print(f"F1-Score Weighted:   {f1_weighted:.4f}")

print(f"\nSố ảnh xử lý lỗi: {len(failed_images)}")
print("============================================")


#Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\n========== CONFUSION MATRIX ==========")
print("Thứ tự nhãn: cat = 0, dog = 1")
print(cm)
print("Hàng: nhãn thực tế | Cột: nhãn dự đoán")
print("======================================")

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix - CNN Cat and Dog Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Đã lưu Confusion Matrix: {CONFUSION_MATRIX_FILE}")


#Classification report

report_dict = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

report_df.to_csv(
    CLASSIFICATION_REPORT_FILE,
    encoding="utf-8-sig"
)

print(
    f"Đã lưu Classification Report: "
    f"{CLASSIFICATION_REPORT_FILE}"
)


#Lưu kết quả
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nĐã lưu kết quả dự đoán: {OUTPUT_FILE}")


#Lưu danh sách ảnh lỗi
if failed_images:
    pd.DataFrame(failed_images).to_csv(
        FAILED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Đã lưu danh sách ảnh lỗi: {FAILED_FILE}")
elif FAILED_FILE.exists():
    #Xóa file cũ nếu lần chạy hiện tại không còn ảnh lỗi
    FAILED_FILE.unlink()

print("\nHoàn tất đánh giá mô hình CNN.")