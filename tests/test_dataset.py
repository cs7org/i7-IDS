from ids_expt.data.dataset import DFDataSet, DataSetConfig, SamplingMethod
import pandas as pd
import os
from pathlib import Path


def sample_dataframe():
    """
    Fixture to create a sample DataFrame for testing.
    """
    data = {
        "feature1": [1, 2, 3, 4, 5] * 60,
        "feature2": [5, 4, 3, 2, 1] * 60,
        "feature3": [10, 20, 30, 40, 50] * 60,
        "Label": ["A", "B", "A", "B", "C"] * 60,
    }
    return pd.DataFrame(data, columns=data.keys())


def test_sampling():
    """
    Fixture to create a sample DataFrame for testing random sampling.
    """
    df = sample_dataframe()
    csv_path = "tests/test_data.csv"
    df.to_csv(csv_path, index=False)
    lbl_counts = df["Label"].value_counts().to_dict()
    config = DataSetConfig(
        csv_path=Path(csv_path),
        label_column="Label",
        labels=["A", "B", "C"],
        features=["feature1", "feature2", "feature3"],
        normal_label="A",
        sampling_method=SamplingMethod.NONE,
        random_state=42,
    )
    ds1 = DFDataSet(config=config)
    ds1.config.csv_path = csv_path
    train_ds, test_ds = ds1.get_datasets()

    test_lbls = test_ds.data.Label.value_counts().to_dict()
    train_lbls = train_ds.data.Label.value_counts().to_dict()

    ds2 = DFDataSet(config=config)
    train_ds2, test_ds2 = ds2.get_datasets()

    # delete csv file after test
    os.remove(csv_path)

    # train df from ds1 and ds2 should be equal
    assert train_ds.data.equals(train_ds2.data), "Train datasets are not equal"

    # test df from ds1 and ds2 should be equal
    assert test_ds.data.equals(test_ds2.data), "Test datasets are not equal"

    # sum of train and test labels should be equal to original label counts
    for label, count in lbl_counts.items():
        assert (
            train_lbls.get(label, 0) + test_lbls.get(label, 0) == count
        ), f"Label counts do not match for {label}"


# test oversampling
def test_oversampling():
    """
    Test to ensure that oversampling is correctly applied to the dataset.
    """
    df = sample_dataframe()
    csv_path = "tests/test_data.csv"
    df.to_csv(csv_path, index=False)
    label_counts = df["Label"].value_counts().to_dict()
    config = DataSetConfig(
        csv_path=Path(csv_path),
        label_column="Label",
        labels=["A", "B", "C"],
        features=["feature1", "feature2", "feature3"],
        normal_label="A",
        sampling_method=SamplingMethod.OVERSAMPLE,
        random_state=42,
    )
    ds = DFDataSet(config=config)
    train_ds, val_ds = ds.get_datasets()
    train_labels = train_ds.data.Label.value_counts().to_dict()
    val_labels = val_ds.data.Label.value_counts().to_dict()

    # delete csv file after test
    os.remove(csv_path)

    assert len(train_ds.data) + len(val_ds.data) > len(
        df
    ), "Oversampling did not increase dataset size"

    for label, count in label_counts.items():
        assert (
            train_labels.get(label, 0) + val_labels[label] >= count
        ), f"Label {label} was not oversampled correctly in training set"


def test_undersampling():
    """
    Test to ensure that undersampling is correctly applied to the dataset.
    """
    df = sample_dataframe()
    csv_path = "tests/test_data.csv"
    df.to_csv(csv_path, index=False)
    label_counts = df["Label"].value_counts().to_dict()

    config = DataSetConfig(
        csv_path=Path(csv_path),
        label_column="Label",
        labels=["A", "B", "C"],
        features=["feature1", "feature2", "feature3"],
        normal_label="A",
        sampling_method=SamplingMethod.UNDERSAMPLE,
        random_state=42,
    )
    ds = DFDataSet(config=config)
    train_ds, val_ds = ds.get_datasets()

    train_labels = train_ds.data.Label.value_counts().to_dict()
    val_labels = val_ds.data.Label.value_counts().to_dict()

    # delete csv file after test
    os.remove(csv_path)

    assert len(train_ds.data) + len(val_ds) < len(
        df
    ), "Undersampling did not decrease dataset size"

    for label, count in label_counts.items():
        assert (
            train_labels.get(label, 0) + val_labels[label] <= count
        ), f"Label {label} was not undersampled correctly in training set"


def test_features():
    """
    Test to ensure that the features are correctly set in the dataset.
    """
    df = sample_dataframe()
    csv_path = "tests/test_data.csv"
    df.to_csv(csv_path, index=False)
    config = DataSetConfig(
        csv_path=Path(csv_path),
        label_column="Label",
        labels=["A", "B", "C"],
        features=["feature1", "feature2"],
        normal_label="A",
        sampling_method=SamplingMethod.NONE,
        random_state=42,
    )
    ds = DFDataSet(config=config)
    train_ds, _ = ds.get_datasets()

    # delete csv file after test
    os.remove(csv_path)

    assert set(train_ds.data.columns) == set(
        config.features + [config.label_column]
    ), "Features are not correctly set in the dataset"


def test_label_encoding():
    """Test to ensure that label encoding is correctly applied."""

    df = sample_dataframe()
    csv_path = "tests/test_data.csv"
    df.to_csv(csv_path, index=False)
    config = DataSetConfig(
        csv_path=Path(csv_path),
        label_column="Label",
        labels=["A", "B", "C"],
        features=["feature1", "feature2", "feature3"],
        normal_label="A",
        sampling_method=SamplingMethod.NONE,
        random_state=42,
    )
    ds = DFDataSet(config=config)
    train_ds, _ = ds.get_datasets()

    # delete csv file after test
    os.remove(csv_path)

    assert set(ds.label_encoding.keys()) == set(
        config.labels
    ), "Label encoding is incorrect"
