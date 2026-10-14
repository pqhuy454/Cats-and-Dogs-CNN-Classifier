from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay


#Cấu hình đường dẫn
PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_DIR / "results" / "prediction_results.csv"
OUTPUT_IMAGE = PROJECT_DIR / "results" / "calibration_curve_cat_dog.png"

N_BINS = 10


#Đọc và kiểm tra dữ liệu
if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file kết quả: {CSV_FILE}\n"
        "Hãy chạy predict.py trước."
    )

df = pd.read_csv(CSV_FILE)

required_columns = {"true_label", "probability"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"CSV thiếu các cột bắt buộc: {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("File CSV không có dữ liệu.")

#Chuyển đổi dữ liệu về dạng số
df["true_label"] = pd.to_numeric(
    df["true_label"], errors="coerce"
)
df["probability"] = pd.to_numeric(
    df["probability"], errors="coerce"
)

#Loại bỏ các dòng không hợp lệ
df = df.dropna(
    subset=["true_label", "probability"]
).copy()

if df.empty:
    raise ValueError("Không có dữ liệu hợp lệ để đánh giá.")

#Kiểm tra nhãn thật
if not df["true_label"].isin([0, 1]).all():
    raise ValueError("true_label chỉ được chứa giá trị 0 hoặc 1.")

#Kiểm tra xác suất
if not df["probability"].between(0, 1).all():
    raise ValueError("probability phải nằm trong khoảng [0, 1].")

y_true = df["true_label"].astype(int).values
y_prob = df["probability"].values

if len(set(y_true)) < 2:
    raise ValueError(
        "Cần có cả ảnh mèo và ảnh chó trong tập đánh giá."
    )


#Vẽ biểu đồ
figure, axes = plt.subplots(figsize=(10, 7))

CalibrationDisplay.from_predictions(
    y_true=y_true,
    y_prob=y_prob,
    n_bins=N_BINS,
    strategy="quantile",
    name="CNN",
    ax=axes,
    color="blue",
    linewidth=3
)

axes.set_title(
    "Calibration Curve - Cats vs Dogs CNN",
    fontsize=16,
    pad=20
)

axes.set_xlabel("Mean predicted probability")
axes.set_ylabel("Fraction of positives")

axes.grid(True, alpha=0.5)

figure.tight_layout()


#Lưu và hiển thị biểu đồ
OUTPUT_IMAGE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

print(f"Số ảnh được đánh giá: {len(df)}")
print(f"Calibration curve đã lưu tại: {OUTPUT_IMAGE}")

plt.show()
plt.close(figure)