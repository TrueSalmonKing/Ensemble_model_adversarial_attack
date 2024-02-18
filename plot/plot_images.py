from secml.figure import CFigure


def plot_images(examples, adv_examples):
    suc_adv_num = 10 if len(adv_examples) > 10 else len(adv_examples)
    fig = CFigure(height=3*2, width=suc_adv_num*2)
    dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(suc_adv_num):
        img_normal = examples[i][0].tondarray().reshape((3,32,32)).transpose(1,2,0)
        img_adv = adv_examples[i][0].tondarray().reshape((3,32,32)).transpose(1,2,0)

        diff_img = img_normal - img_adv
        diff_img -= diff_img.min()
        diff_img /= diff_img.max()

        fig.subplot(3,suc_adv_num,i+1)
        fig.sp.imshow(img_normal)
        fig.sp.title('{0}'.format(dataset_labels[examples[i][1].item()]))
        fig.sp.xticks([])
        fig.sp.yticks([])

        fig.subplot(3,suc_adv_num,i+(1+suc_adv_num))
        fig.sp.imshow(img_adv)
        fig.sp.title('{0}'.format(dataset_labels[adv_examples[i][1][0].item()]))
        fig.sp.xticks([])
        fig.sp.yticks([])

        fig.subplot(3,suc_adv_num,i+(1+2*suc_adv_num))
        fig.sp.imshow(diff_img)
        fig.sp.title('Amplified perturbation')
        fig.sp.xticks([])
        fig.sp.yticks([])
    fig.tight_layout()
    fig.show()
