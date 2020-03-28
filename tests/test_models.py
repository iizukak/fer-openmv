from tensorflow import keras
import models


def test_fer_small():
    model = models.fer_small()
    assert type(model) == keras.Model
    assert model.count_params() == 223143


def test_resnet_50():
    model = models.resnet_50()
    assert type(model) == keras.Model
    assert model.count_params() == 24107783
