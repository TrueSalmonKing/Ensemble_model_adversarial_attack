from load_models import load_cifar10_linf_models
from secml.data.loader.c_dataloader_cifar import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
import random
from adv_attacks.multi_model_untargeted_attack import multi_model_untargeted_attack, get_X, get_Y
from adv_attacks.fgsm_untargeted import fgsm_untargeted
from secml.core.constants import inf
from plot.plot_images import plot_images
from transferability_test import trans_test
from plot.plot_trans import plot_trans

model_names = ['Ding2020MMA', 'Zhang2019You', 'Gowal2020Uncovering_70_16']

target_model_names = ['Peng2023Robust', 'Wang2023Better_WRN-70-16',
                      'Cui2023Decoupled_WRN-28-10', 'Wang2023Better_WRN-28-10',
                      'Rebuffi2021Fixing_70_16_cutmix_extra',
                      'Gowal2021Improving_70_16_ddpm_100m',
                      'Gowal2020Uncovering_70_16_extra']

"""
    We Consider 11 models from RobustBench (CIFAR10, L-inf)

    1 Standard WideRestNet model
    3 Least robust models with norm=Linf and eps=8/255 with differing archs:
      WideResNet-28-4
      WideResNet-34-10
      WideResNet-70-16

"""

# We load the robustbench models inside MLSec

secml_models = load_cifar10_linf_models(model_names)
standard_secml_models = load_cifar10_linf_models(['Standard'])

eps = 8/255

# We load our dataset and choose 1000 images from the test set
train_ds, test_ds = CDataLoaderCIFAR10().load()
samples_num = 1000
adv_example_num = samples_num

# We randomize the chosen samples
seed_value = 666
random.seed(seed_value)
indx = random.sample(range(10000), samples_num)
test_ds = test_ds[indx, :]
normalizer = CNormalizerMinMax().fit(train_ds.X)
test_ds.X = normalizer.transform(test_ds.X)

# We generate adversarial examples for both:
# The standard model
c_examples, c_adv_examples, c_adv_ds = multi_model_untargeted_attack(fgsm_untargeted, inf, test_ds,
                                                                     standard_secml_models, adv_example_num, eps)
# The ensemble of models
examples, adv_examples, adv_ds = multi_model_untargeted_attack(fgsm_untargeted, inf, test_ds, secml_models,
                                                               adv_example_num, eps)

print(f"{len(adv_examples)} of the {len(adv_ds)} adversarial samples cause a misclassification")

plot_images(examples, adv_examples)
plot_images(c_examples, c_adv_examples)

"""
    We Consider 7 models from RobustBench (CIFAR10, L-inf) to test the
    transferability

    7 most robust models with norm=Linf and eps=8/255

"""

target_secml_models = load_cifar10_linf_models(target_model_names)

adv_ds_X = get_X(adv_ds)
c_adv_ds_X = get_X(c_adv_ds)


c_origin_error, c_trans_error, c_transfer_rate = trans_test(target_model_names,
                                                            target_secml_models,
                                                            test_ds.X,
                                                            test_ds.Y,
                                                            c_adv_ds_X)

origin_error, trans_error, transfer_rate = trans_test(target_model_names,
                                                      target_secml_models,
                                                      test_ds.X,
                                                      test_ds.Y,
                                                      adv_ds_X)

print(transfer_rate)

plot_trans(target_model_names, target_secml_models, c_trans_error, c_transfer_rate)
plot_trans(target_model_names, target_secml_models, trans_error, transfer_rate)
c_transfer_rate = 0.1045

print(f"An increase in the aversage test error rate of {(transfer_rate-c_transfer_rate)*100:.2f}% when using 3 models "
      f"instead of one standard model to compute the adversarial examples")

"""
    We test the transferability of the successful adversarial examples
    generated for 7 most robust models with norm=Linf and eps=8/255

"""

adv_examples_X = get_X(adv_examples)
examples_X = get_X(examples)
examples_Y = get_Y(examples)
c_adv_examples_X = get_X(c_adv_examples)
c_examples_X = get_X(c_examples)
c_examples_Y = get_Y(c_examples)


c_origin_error, c_trans_error, c_transfer_rate = trans_test(target_model_names,
                                                            target_secml_models,
                                                            c_examples_X,
                                                            c_examples_Y,
                                                            c_adv_examples_X)

origin_error, trans_error, transfer_rate = trans_test(target_model_names,
                                                      target_secml_models,
                                                      examples_X,
                                                      examples_Y,
                                                      adv_examples_X)

plot_trans(target_model_names, target_secml_models, c_trans_error, c_transfer_rate)
plot_trans(target_model_names, target_secml_models, trans_error, transfer_rate)

print(f"An increase in the adversarial samples' success rate of {(transfer_rate-c_transfer_rate)*100:.2f}% "
      f"when using 3 models "
      f"instead of one standard model to compute successful adversarial examples")
