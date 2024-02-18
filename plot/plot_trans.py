from secml.figure import CFigure
from secml.array import CArray


def plot_trans(target_model_names, target_model_list, trans_error, transfer_rate):
    trans_acc = CArray(trans_error) * 100  # Show results in percentage

    fig = CFigure(height=1)
    a = fig.sp.imshow(trans_acc.reshape((1, len(target_model_list))),
                      cmap='Oranges', interpolation='nearest',
                      alpha=.65, vmin=60, vmax=70)

    fig.sp.xticks(CArray.arange((len(target_model_list))))
    fig.sp.xticklabels(target_model_names,
                       rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(target_model_list)):
        fig.sp.text(i, 0, trans_acc[i].round(2).item(), va='center', ha='center')

    fig.sp.title("Test error of target classifiers under attack (%)")

    fig.show()

    print("\nAverage transfer rate: {:.2%}".format(transfer_rate))