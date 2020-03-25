from src import dataset
import numpy as np

FIXTURE_PATH = "./tests/fixture.csv"


def test_load_csv():
    lines = dataset._load_csv(FIXTURE_PATH)
    assert len(lines) == 9
    assert len(lines[0]) == 3


def test_parse():
    lines = dataset._load_csv(FIXTURE_PATH)
    (x_train, y_train), (x_public_test, y_public_test), (x_private_test, y_private_test) =\
        dataset._parse(lines)
    assert len(x_train) == len(x_public_test) == len(x_private_test) == 3
    assert len(y_train) == len(y_public_test) == len(y_private_test) == 3

    assert x_train[0].shape == (48, 48)
    assert x_public_test[0].shape == (48, 48)
    assert x_private_test[0].shape == (48, 48)

    assert (y_train == np.array([0, 0, 2], dtype=np.uint8)).all()
    assert (y_public_test == np.array([0, 1, 4], dtype=np.uint8)).all()
    assert (y_private_test == np.array([0, 5, 6], dtype=np.uint8)).all()


def test_load_dataset():
    (x_train, y_train), (x_public_test, y_public_test), (x_private_test, y_private_test) =\
        dataset.load_dataset(FIXTURE_PATH)
    assert len(x_train) == len(x_public_test) == len(x_private_test) == 3
    assert len(y_train) == len(y_public_test) == len(y_private_test) == 3
