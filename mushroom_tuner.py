import tensorflow as tf
import tensorflow_transform as tft
import kerastuner
import os

from collections import namedtuple
from tensorflow.keras import layers
from kerastuner import HyperParameters
from tfx.components.trainer.fn_args_utils import FnArgs

TunerFnResult = namedtuple(
    'TunerFnResult',
    ['tuner', 'fit_kwargs']
)

LABEL_KEY = "class"
CATEGORICAL_FEATURE_KEYS = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=1, batch_size=32):
    """Create dataset from file pattern."""
    if isinstance(file_pattern, list):
        # Gabungkan semua file patterns
        all_files = []
        for pattern in file_pattern:
            files = tf.io.gfile.glob(pattern)
            all_files.extend(files)
        file_pattern = all_files
    
    if not file_pattern:
        raise ValueError("File pattern is empty!")

    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    # Buat dataset dari file patterns
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
        shuffle=True,
        shuffle_buffer_size=1000
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(hp):
    """Build model with hyperparameters."""
    inputs = {}
    encoded_inputs = []

    for key in CATEGORICAL_FEATURE_KEYS:
        feat_key = transformed_name(key)
        inputs[feat_key] = tf.keras.Input(shape=(1,), name=feat_key, dtype=tf.int64)
        embed = layers.Embedding(input_dim=20, output_dim=4)(inputs[feat_key])
        flat = layers.Flatten()(embed)
        encoded_inputs.append(flat)

    x = tf.keras.layers.concatenate(encoded_inputs)
    x = layers.Dense(hp.Int("units_1", 32, 128, step=16), activation='relu')(x)
    x = layers.Dense(hp.Int("units_2", 16, 128, step=16), activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [0.001, 0.01, 0.1])),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def tuner_fn(fn_args: FnArgs):
    """Tuner function for hyperparameter optimization."""
    print("=== DEBUG LOG: Masuk ke tuner_fn ===")
    print("train_files:", fn_args.train_files)
    print("eval_files:", fn_args.eval_files)
    print("transform_graph_path:", fn_args.transform_graph_path)
    print("working_dir:", fn_args.working_dir)

    if not fn_args.train_files or not fn_args.eval_files or not fn_args.transform_graph_path:
        raise ValueError("Salah satu dari train_files / eval_files / transform_graph_path kosong!")

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Buat dataset untuk training dan validation
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=None, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=None, batch_size=32)

    # Debug: Cek apakah dataset bisa dibaca
    try:
        train_sample = next(iter(train_dataset.take(1)))
        eval_sample = next(iter(eval_dataset.take(1)))
        print("DEBUG: train_dataset sample shape:", {k: v.shape for k, v in train_sample[0].items()})
        print("DEBUG: eval_dataset sample shape:", {k: v.shape for k, v in eval_sample[0].items()})
        print("DEBUG: train_dataset label shape:", train_sample[1].shape)
        print("DEBUG: eval_dataset label shape:", eval_sample[1].shape)
    except Exception as e:
        print("ERROR saat membaca dataset:", e)
        raise

    # Inisialisasi tuner
    tuner = kerastuner.RandomSearch(
        build_model,
        max_trials=5,  # Kurangi untuk testing
        objective='val_binary_accuracy',
        directory=fn_args.working_dir,
        project_name='mushroom_tuning',
        max_consecutive_failed_trials=5  # Tambahkan tolerance untuk trial yang gagal
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": 50,
            "validation_steps": 25,
            "epochs": 3,  # Kurangi untuk testing
            "verbose": 1
        }
    )
