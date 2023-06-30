"""Metadata for the Chinese MNIST dataset."""
import text_recognizer.metadata.shared as shared

DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME

DIMS = (1, 64, 64)
OUTPUT_DIMS = (1,)
MAPPING = list(range(15))

TRAIN_SIZE = 14000
VAL_SIZE = 1000
