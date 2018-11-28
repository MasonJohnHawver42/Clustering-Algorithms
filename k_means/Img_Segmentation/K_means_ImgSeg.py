import numpy as np
import k_means.K_means as KM

def img_seg(img_a, k_points, max_gen):

    array = np.reshape(img_a, (img_a.shape[0] * img_a.shape[1], img_a.shape[2]))

    test = KM.k_means(array, k_points, max_gen)


    new_array = np.zeros(array.shape).astype(np.int64)

    for i in range(test[0].shape[0]):
        new_array += ((np.nan_to_num(test[1][i] / test[1][i])).astype(np.int64) * test[0][i].astype(np.int64))   #  (np.tile((test[0][i]).astype(np.int64), test[1][i].shape[0]).reshape(test[1][i].shape)).astype(np.int64))

    new_img_a = np.reshape(new_array, img_a.shape).astype(np.uint8)

    return new_img_a