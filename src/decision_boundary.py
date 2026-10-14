from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Cấu hình đường dẫn
PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_DIR / "results" / "prediction_results.csv"
OUTPUT_FILE = PROJECT_DIR / "results" / "decision_boundary.png"


#Đọc dữ liệu
if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Chưa có file kết quả: {CSV_FILE}\n"
        "Hãy chạy predict.py trước."
    )

df = pd.read_csv(CSV_FILE)

required_columns = {"predicted_class", "probability"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"File CSV thiếu các cột: {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("File CSV không có dữ liệu dự đoán.")

df["probability"] = pd.to_numeric(
    df["probability"], errors="coerce"
)

df = df.dropna(subset=["probability"]).reset_index(drop=True)

if df.empty:
    raise ValueError("Không có xác suất dự đoán hợp lệ.")

if not df["probability"].between(0, 1).all():
    raise ValueError("Xác suất dự đoán phải nằm trong khoảng [0, 1].")


#Thống kê kết quả
counts = df["predicted_class"].value_counts()

num_dog = int(counts.get("dog", 0))
num_cat = int(counts.get("cat", 0))

print("========== THỐNG KÊ DỰ ĐOÁN ==========")
print(f"Chó (dog): {num_dog}")
print(f"Mèo (cat): {num_cat}")
print(f"Tổng ảnh: {len(df)}")
print("======================================")


#Vẽ biểu đồ
plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=df,
    x=df.index,
    y="probability",
    hue="predicted_class",
    hue_order=["cat", "dog"],
    palette={"dog": "blue", "cat": "red"},
    s=50,
    alpha=0.75
)

#Ranh giới quyết định
plt.axhline(
    y=0.5,
    color="green",
    linestyle="--",
    linewidth=2,
    label="Decision Boundary (0.5)"
)

#Vùng xác suất dự đoán
plt.fill_between(
    x=[-1, len(df)],
    y1=0.5,
    y2=1.1,
    color="blue",
    alpha=0.05
)

plt.fill_between(
    x=[-1, len(df)],
    y1=-0.1,
    y2=0.5,
    color="red",
    alpha=0.05
)

plt.title(
    "Phân bố dự đoán của mô hình CNN (Decision Boundary)",
    fontsize=14
)
plt.xlabel("Thứ tự ảnh", fontsize=12)
plt.ylabel("Xác suất dự đoán lớp chó - P(dog)", fontsize=12)

plt.ylim(-0.1, 1.1)
plt.xlim(-1, len(df))
plt.grid(True, alpha=0.3)
plt.legend(loc="center right")
plt.tight_layout()


#Lưu và hiển thị biểu đồ
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nĐã lưu biểu đồ tại: {OUTPUT_FILE}")

plt.show()
plt.close()