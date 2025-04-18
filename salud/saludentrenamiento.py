import os
import warnings
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, DenseNet121, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Silenciar warnings y logs innecesarios
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("📦 Dispositivos disponibles:")
print(tf.config.list_physical_devices())

# === CONFIGURACIÓN ===
dataset_path = r'C:\Users\isard\Downloads\FOTOS\lung_colon_image_set\TODO'  # carpeta con todas las clases
img_size = 768
batch_size = 8
epochs = 10  # puedes subirlo luego si ves que no hay overfitting
model_dir = r'C:\ruta\models_multiclase'
os.makedirs(model_dir, exist_ok=True)

# === MODELOS A ENTRENAR ===
model_zoo = {
    'EfficientNetB3': EfficientNetB3,
    'DenseNet121': DenseNet121,
    'InceptionV3': InceptionV3
}

# === CARGA DE DATOS ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())
print(f"\n📊 Clases detectadas ({num_classes}): {class_names}")

# === CONSTRUCCIÓN DEL MODELO ===
def build_model(model_fn, input_shape=(768, 768, 3), num_classes=6):
    base = model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False  # Congelamos la base

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base.input, outputs=output)

# === ENTRENAMIENTO ===
for model_name, model_fn in model_zoo.items():
    print(f"\n🧠 Entrenando modelo: {model_name}")

    model = build_model(model_fn, input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model_path = os.path.join(model_dir, f'{model_name}_multiclase.h5')
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    val_loss, val_acc = model.evaluate(val_gen)
    print(f"✅ {model_name} - Precisión validación: {val_acc:.4%}\n")