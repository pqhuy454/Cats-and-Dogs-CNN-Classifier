from pathlib import Path
import json
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Cấu hình đường dẫn
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "data" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_PATH = MODEL_DIR / "model_CNN.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.json"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

#Tạo thư mục lưu mô hình và kết quả
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


#Kiểm tra dữ liệu
if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu train: {TRAIN_DIR}\n"
        "Hãy chạy clean_dataset.py và prepare_dataset.py trước."
    )

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu validation: {VALIDATION_DIR}"
    )


#Chỉ tăng cường dữ liệu train
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

#Validation chỉ chuẩn hóa pixel, không tăng cường dữ liệu
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    directory=str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["cat", "dog"],
    shuffle=True,
    seed=SEED
)

validation_generator = validation_datagen.flow_from_directory(
    directory=str(VALIDATION_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["cat", "dog"],
    shuffle=False
)

if train_generator.samples == 0:
    raise ValueError("Tập train không có ảnh.")

if validation_generator.samples == 0:
    raise ValueError("Tập validation không có ảnh.")

print("\nPhân bố nhãn:", train_generator.class_indices)
print("Số ảnh train:", train_generator.samples)
print("Số ảnh validation:", validation_generator.samples)


#Xây dựng mô hình
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    #Block 1: Đặc trưng cơ bản
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    #Block 2: Đặc trưng trung gian
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    #Block 3: Đặc trưng phức tạp
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    #Phân loại
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])


#Biên dịch mô hình
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
)

model.summary()


#Callbacks
callbacks = [
    #Lưu mô hình có validation loss tốt nhất
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    #Dừng sớm nếu mô hình không còn cải thiện
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


#Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)


#Lưu lịch sử huấn luyện
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(HISTORY_PATH, "w", encoding="utf-8") as file:
    json.dump(history_data, file, ensure_ascii=False, indent=4)

print("\nHuấn luyện hoàn tất!")
print(f"Mô hình: {MODEL_PATH}")
print(f"Lịch sử huấn luyện: {HISTORY_PATH}")