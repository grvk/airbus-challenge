import torch

from src.models.fcn8s import FCN8s

net = FCN8s(2, initialize="imagenet")
# add more tests
