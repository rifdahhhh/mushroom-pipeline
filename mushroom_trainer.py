
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os

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

def input_fn(file_pattern, tf_transform_output, num_epochs=10, batch_size=64):
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
        shuffle=True)
    return dataset

def model_builder():
    inputs = {}
    encoded_inputs = []

    # Buat input dan embedding untuk semua fitur kategorikal
    for key in CATEGORICAL_FEATURE_KEYS:
        feat_key = transformed_name(key)
        inputs[feat_key] = tf.keras.Input(shape=(1,), name=feat_key, dtype=tf.int64)
        vocab_size = 20  # asumsi maksimum kategori per fitur
        embed = tf.keras.layers.Embedding(vocab_size, 4)(inputs[feat_key])
        flat = tf.keras.layers.Flatten()(embed)
        encoded_inputs.append(flat)

    x = tf.keras.layers.concatenate(encoded_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    model = model_builder()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs'))
    
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=100,
        validation_steps=50,
        epochs=5,
        callbacks=[tensorboard_callback]
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
