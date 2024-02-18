from secml.array import CArray

"""
  Function aimed to perform ensemble model adversarial attacks on a dataset test_ds
  With the addition of the ability to return after a number of adversarial
  examples that cause a misclassfication is reached using adv_example_num
"""


def multi_model_untargeted_attack(algo, norm, test_ds, clfs, adv_example_num, eps, alpha=0.1, steps=100):
    adv_ds = []
    adv_examples = []
    examples = []

    j = 0
    k = 0

    for i in range(adv_example_num):
        k += 1
        x0, y0 = test_ds[i, :].X, test_ds[i, :].Y

        # print(f"Starting point has label: {y0.item()}")
        # print('generating sample'+str(j))
        if algo.__name__ == 'fgsm_untargeted':
            x_adv, ys_adv, attack_path = algo(x0, y0, clfs, eps, norm)
        elif algo.__name__ == 'pgd_untargeted':
            x_adv, ys_adv, attack_path = algo(x0, y0, clfs, eps, alpha, steps, norm)
        if y0.item() not in [y_adv.item() for y_adv in ys_adv]:
            # print(f"adversarial sample with starting label {y0.item()} generated successfully,
            # with labels : {[y.item() for y in ys_adv]}")
            examples.append([x0, y0])
            adv_examples.append([x_adv, ys_adv, attack_path])
            j += 1
        adv_ds.append([x_adv, ys_adv, attack_path])

        if (i+1)%(test_ds.num_samples/10) == 0:
            print(f"generation progress: {100*(i+1)/test_ds.num_samples}%")

        if j >= adv_example_num:
            return examples, adv_examples, adv_ds
    return examples, adv_examples, adv_ds


"""
  Helper functions to get the image samples and the image labels
"""

def get_X(ds):
    ds_X = None

    for sample in ds:
        if ds_X is not None:
            ds_X = ds_X.append(sample[0], axis=0)
        if ds_X is None:
            ds_X = CArray(sample[0])

    return ds_X


def get_Y(ds):
    ds_Y = None

    for sample in ds:
        if ds_Y is not None:
            ds_Y = ds_Y.append(sample[1], axis=0)
        if ds_Y is None:
            ds_Y = CArray(sample[1])

    return ds_Y
