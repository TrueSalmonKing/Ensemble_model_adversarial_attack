# Metric to use for testing transferability
from secml.ml.peval.metrics import CMetricTestError


def trans_test(target_model_names, target_model_list, test_ds_x, test_ds_y, adv_ds_x):
    metric = CMetricTestError()

    origin_error = []
    trans_error = []
    transfer_rate = 0.0

    for target_clf_name, target_clf in zip(target_model_names, target_model_list):

        print("\nTesting transferability of {:}".format(target_clf_name))

        origin_error_clf = metric.performance_score(
                y_true=test_ds_y, y_pred=target_clf.predict(test_ds_x))
        origin_error.append(origin_error_clf)

        trans_error_clf = metric.performance_score(
            y_true=test_ds_y, y_pred=target_clf.predict(adv_ds_x))

        trans_error.append(trans_error_clf)
        transfer_rate += trans_error_clf

    # Computing the transfer rate
    transfer_rate /= len(target_model_list)

    return origin_error, trans_error, transfer_rate
