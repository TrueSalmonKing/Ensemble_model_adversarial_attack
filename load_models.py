from robustbench.utils import load_model
from secml.ml.classifiers import CClassifierPyTorch

"""
  We define a function to aid in loading several cifar10/Linf norm models
  into secml
"""


def load_cifar10_linf_models(target_model_names):
    models = []
    for model_name in target_model_names:
        models.append(load_model(model_name=model_name, dataset='cifar10', threat_model='Linf'))

    secml_models = []
    for model in models:
        secml_models.append(CClassifierPyTorch(model, input_shape=(3, 32, 32), pretrained=True))

    return secml_models
