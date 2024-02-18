from secml.ml.classifiers.loss import CLossCrossEntropy
from secml.array import CArray
from secml.core.constants import inf


def fgsm_untargeted(x, y, clfs, eps, norm=inf):
    loss_func = CLossCrossEntropy()

    avg_score = sum([clf.decision_function(x) for clf in clfs]) / len(clfs)

    # Gradient of the loss wrt the clf logits
    loss_gradients = loss_func.dloss(y_true=y, score=avg_score)
    # Gradient of the clf logits wrt the input
    clf_gradients = [clf.grad_f_x(x, y) for clf in clfs]
    # Compute total gradient as a mean of the gradients across the models
    avg_gradient = sum(clf_gradients) / len(clf_gradients)

    # Chain rule
    total_gradient = avg_gradient * loss_gradients

    # Normalize the gradient (takes only the direction and discards the magnitude)
    if total_gradient.norm() != 0:
        total_gradient /= total_gradient.norm()

    # Make a step
    x_adv = x + eps * total_gradient.sign()

    # We project inside epsilon-ball
    delta = x_adv - x
    if delta.norm(norm) > eps:
        delta = delta / delta.norm(norm) * eps
        x_adv = x + delta

    # We force input bounds
    x_adv = x_adv.clip(0, 1)

    path = CArray.zeros((2, x.shape[1]))
    path[0, :] = x.deepcopy()
    path[1, :] = x_adv

    return x_adv, [clf.predict(x_adv) for clf in clfs], path
