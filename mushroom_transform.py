
import tensorflow as tf
import tensorflow_transform as tft


# Nama kolom label (target)
LABEL_KEY = "class"

# Daftar semua fitur kategorikal (selain label)
CATEGORICAL_FEATURE_KEYS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

def transformed_name(key):
    """Utility function to rename transformed features."""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs: Dictionary of input feature tensors.

    Returns:
        Dictionary of transformed feature tensors.
    """
    outputs = {}

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Perbaikan: pastikan output label bertipe int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.where(tf.equal(inputs[LABEL_KEY], 'p'), 1, 0),
        tf.int64
    )

    return outputs
