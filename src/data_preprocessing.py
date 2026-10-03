"""Dataset loading and reproducible train/test splitting."""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_dataset(as_frame=True):
    """Load California Housing data; downloads/cache handled by scikit-learn."""
    bunch = fetch_california_housing(as_frame=as_frame)
    if as_frame:
        X = bunch.data.copy()
        y = bunch.target.copy()
    else:
        X, y = bunch.data, bunch.target
    return X, y, list(bunch.feature_names)


def split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target reproducibly."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
