#####
#####
#####
#####
#####


from show_3d_images import show_3d_images
import numpy as np
import copy
import skimage
import skimage.measure


def dsc(mask_gt, mask_pred):
    # Compute dice similarity score for a single image slice
    intersection = np.logical_and(mask_gt, mask_pred)
    volume_sum = mask_gt.sum() + mask_pred.sum()
    dsc = 2 * intersection.sum() / volume_sum
    return dsc


def scale_to_0_1(mat):
    # Scale image between 0 and 1
    mat = mat - np.amin(mat)
    if np.amax(mat) != 0:
        mat = mat / np.amax(mat)
    return mat


def iou(mask_gt, mask_pred):
    # Compute intersection over union for a single image slice
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def get_largest_2D_conn(mask_input):
    # NOTE: THIS STEP BELOW IS SUPER IMPORTANT
    # THIS FUNCTION TAKES A 2D MASK FILE AND
    # RETURNS ONLY THE LARGEST CONNECTED STRUCTURE
    #           SO IF THE NETWOKR SEGMENTS THE TIBIA BUT ALSO
    # LEAVES A FEW ERRONEOUS PIXELS THROUGHOUT THE IMAGE, THE
    #       TIBIA WILL HAVE MORE PIXELS TOUCHING EACH OTHER SO IT WILL
    # REUTRN ONLY THAT ONE STRUCTURE

    mask = copy.deepcopy(mask_input)

    (batch_dim, channels, n_rows, n_cols) = mask.shape

    for iter_scan in range(batch_dim):

        all_labels = skimage.measure.label(mask[iter_scan, :, :, :].squeeze())

        max_label = np.amax(all_labels)

        # print(all_labels)
        # print(all_labels.shape)
        # print(max_label)

        if max_label > 1:
            tmp1 = mask[iter_scan, :, :, :].reshape((n_rows, n_cols, 1))
            # show_3d_images(tmp1)
            max_inds = 0
            max_label_inds = 0
            for iter_label in range(max_label):
                current_inds = np.count_nonzero(all_labels == (iter_label + 1))
                # print(current_inds)
                if current_inds > max_label_inds:
                    max_inds = iter_label + 1
                    max_label_inds = current_inds

            tmp2 = all_labels == (max_inds)
            tmp2 = tmp2.reshape((n_rows, n_cols, 1))
            # show_3d_images(np.hstack((tmp1,tmp2)))
            mask[iter_scan, :, :, :] = tmp2.squeeze()

    return mask


def get_dsc_iou(
    mask_gt, mask_pred, return_all_vals=False, only_largest_structure=False
):
    # This is used to call the DSC and IOU functions. Iterates through
    # entire batch and computes them per-image-slice

    (batch_dim, channels, n_rows, n_cols) = mask_gt.shape

    mask_gt = mask_gt > 0.5
    mask_pred = mask_pred > 0.5

    if only_largest_structure:

        mask_pred = get_largest_2D_conn(mask_pred)

    iou_vals = []
    dsc_vals = []

    for iter_scan in range(batch_dim):

        tmp_iou = iou(
            mask_gt=mask_gt[iter_scan, :, :, :].squeeze(),
            mask_pred=mask_pred[iter_scan, :, :, :].squeeze(),
        )

        iou_vals.append(tmp_iou)

        tmp_dsc = dsc(
            mask_gt=mask_gt[iter_scan, :, :, :].squeeze(),
            mask_pred=mask_pred[iter_scan, :, :, :].squeeze(),
        )

        dsc_vals.append(tmp_dsc)

    iou_mean = np.asarray(iou_vals).mean()
    dsc_mean = np.asarray(dsc_vals).mean()

    if return_all_vals:
        return iou_mean, dsc_mean, iou_vals, dsc_vals

    else:
        return iou_mean, dsc_mean


def erode_by_one_pixel(mask_input):
    # Useful for morphological erosion operation
    mask = copy.deepcopy(mask_input)

    (batch_dim, channels, n_rows, n_cols) = mask.shape
    for iter_slice in range(batch_dim):
        tmp = scipy.ndimage.morphology.binary_erosion(
            input=mask[iter_slice, :, :].squeeze()
        )
        tmp2 = np.hstack((mask[iter_slice, :, :].squeeze(), tmp))
        # show_3d_images(tmp2)
        mask[iter_slice, :, :] = tmp
    return mask


def prep_images_for_plot(input_image, predicted_mask, ground_truth):

    # Important code for viewing images that works but is some of the dumbest hacks
    # I've written.
    # Because the input image can change in shape, this
    # Just horizontally stacks them in correct order based on number of input dims

    predicted_mask = predicted_mask.astype(np.float32)
    (batch_dim, n_channel, n_rows, n_cols) = input_image.shape

    if batch_dim > 1:
        predicted_mask = predicted_mask.squeeze()
        ground_truth = ground_truth.squeeze()
        if n_channel == 1:
            tmp1 = input_image.squeeze().transpose((1, 2, 0))
            tmp2 = predicted_mask.transpose((1, 2, 0))
            tmp3 = ground_truth.transpose((1, 2, 0))
            for iter_slice in range(batch_dim):
                tmp1[:, :, iter_slice] = scale_to_0_1(tmp1[:, :, iter_slice])
                tmp2[:, :, iter_slice] = scale_to_0_1(tmp2[:, :, iter_slice])
                tmp3[:, :, iter_slice] = scale_to_0_1(tmp3[:, :, iter_slice])
            img = np.hstack((tmp1, tmp2, tmp3))
        elif n_channel == 2:
            tmp1 = input_image[:, 0, :, :].squeeze().transpose((1, 2, 0))
            tmp2 = input_image[:, 1, :, :].squeeze().transpose((1, 2, 0))
            tmp3 = predicted_mask.transpose((1, 2, 0))
            tmp4 = ground_truth.transpose((1, 2, 0))
            for iter_slice in range(batch_dim):
                tmp1[:, :, iter_slice] = scale_to_0_1(tmp1[:, :, iter_slice])
                tmp2[:, :, iter_slice] = scale_to_0_1(tmp2[:, :, iter_slice])
                tmp3[:, :, iter_slice] = scale_to_0_1(tmp3[:, :, iter_slice])
                tmp4[:, :, iter_slice] = scale_to_0_1(tmp4[:, :, iter_slice])
            img = np.hstack((tmp1, tmp2, tmp3, tmp4))
        elif n_channel == 3:
            tmp1 = input_image[:, 0, :, :].squeeze().transpose((1, 2, 0))
            tmp2 = input_image[:, 1, :, :].squeeze().transpose((1, 2, 0))
            tmp3 = input_image[:, 2, :, :].squeeze().transpose((1, 2, 0))
            tmp4 = predicted_mask.transpose((1, 2, 0))
            tmp5 = ground_truth.transpose((1, 2, 0))
            for iter_slice in range(batch_dim):
                tmp1[:, :, iter_slice] = scale_to_0_1(tmp1[:, :, iter_slice])
                tmp2[:, :, iter_slice] = scale_to_0_1(tmp2[:, :, iter_slice])
                tmp3[:, :, iter_slice] = scale_to_0_1(tmp3[:, :, iter_slice])
                tmp4[:, :, iter_slice] = scale_to_0_1(tmp4[:, :, iter_slice])
                tmp5[:, :, iter_slice] = scale_to_0_1(tmp5[:, :, iter_slice])
            img = np.hstack((tmp1, tmp2, tmp3, tmp4, tmp5))

    show_3d_images(img)


def normalize_data_2(TE_1, TE_2):
    # Not used anymore.
    echo1_std = np.std(TE_1.flatten())
    echo1_mean = np.mean(TE_1.flatten())
    TE_1 = scale_to_z_curve(TE_1, echo1_mean, echo1_std)
    TE_2 = scale_to_z_curve(TE_2, echo1_mean, echo1_std)
    return TE_1, TE_2


# def scale_to_z_curve(mat,mean,std):
#     return (mat - mean)/std


def normalize_data(TE_1):
    #
    echo1_std = np.std(TE_1.flatten())
    echo1_mean = np.mean(TE_1.flatten())
    TE_1 = scale_to_z_curve(TE_1, echo1_mean, echo1_std)
    return TE_1, echo1_std, echo1_mean


def scale_to_z_curve(mat, mean=-100, std=-100):
    # Scales image to 0 mean and 1 std
    # unless Mean and Std are specified

    if mean == -100:
        mean = np.mean(mat.flatten())

    if std == -100:
        std = np.std(mat.flatten())

    if np.sum(np.isnan(mat.flatten())):
        tmp = np.zeros(mat.shape)
        return tmp
    else:
        return (mat - mean) / std


def crop_img(img, start_ind, end_ind):
    # .... It literally just crops the image based on the input indices

    start_ind = int(start_ind)
    end_ind = int(end_ind)
    return img[start_ind:end_ind, start_ind:end_ind]


def print_img_characteristics(img):
    # useful in initial debugging/dev phase
    # Prints a lot of image metrics to troubleshoot
    # Data generator issues
    print("IMG SHAPE " + str(img.shape))
    img = img.flatten()
    mean = np.mean(img)
    std = np.std(img)
    val_max = np.amax(img)
    val_min = np.amin(img)

    print("MEAN " + str(mean))
    print("STD " + str(std))
    print("MAX " + str(val_max))
    print("MIN " + str(val_min))
    print("\n\n")
    return
