from tensorflow import keras
from src import resnet


def test_resnet_8():
    model = resnet.resnet_8()
    assert type(model) == keras.Model
    assert model.count_params() == 221895
