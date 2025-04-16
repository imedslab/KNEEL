import cv2
import numpy as np
import torch
import logging
import solt.transforms as slt
import solt

from kneel.utils import read_dicom, process_xray
from kneel.model.ensemble import NFoldInferenceModel



class LandmarkAnnotator(object):
    def __init__(self, models, mean, std, annotator_type, device='cpu', jit_trace=True, logger=None):
        assert annotator_type in ['global_search', 'local_search']
        if logger is None:
            logger = logging.getLogger('Landmark Annotator')

        self.logger = logger
        self.device = torch.device(device)
        self.net = NFoldInferenceModel(models).to(self.device)
        self.net.eval()
        logger.log(logging.INFO, f'Loaded 5 folds inference model to {device}')
        
        if jit_trace:
            logger.log(logging.INFO, 'Optimizing with torch.jit.trace')
            dummy = torch.FloatTensor(2, 3, models[0].crop, models[0].crop).to(device=self.device)
            with torch.no_grad():
                self.net = torch.jit.trace(self.net, dummy)
        self.mean_vector, self.std_vector = mean, std

        self.annotator_type = annotator_type
        self.img_spacing = models[0].spacing

        self.trf = solt.Stream([
                slt.Pad((models[0].pad, models[0].pad), padding='z'),
                slt.Crop((models[0].crop, models[0].crop), crop_mode='c'),
            ])


    def wrap_slt(self, img):
        if self.annotator_type == 'global_search':
            img = np.dstack((img, img, img))
            _, col, _ = img.shape
            # (right, left) encoding
            img = (img[:, :col // 2 + col % 2], img[:, col // 2:])
        else:
            img_right = np.dstack((img[0], img[0], img[0]))
            img_left = np.dstack((img[1], img[1], img[1]))
            img = (img_right, img_left)

        return solt.DataContainer(img, "II")

    @staticmethod
    def pad_img(img, pad):
        if pad is not None:
            if not isinstance(pad, tuple):
                pad = (pad, pad)
            row, col = img.shape
            tmp = np.zeros((row + 2 * pad[0], col + 2 * pad[1]), dtype=img.dtype)
            tmp[pad[0]:pad[0] + row, pad[1]:pad[1] + col] = img
            return tmp
        else:
            return img

    @staticmethod
    def read_dicom(img_path, new_spacing, return_orig=False, pad_img=None):
        res = read_dicom(img_path)
        if res is None:
            return []
        img_orig, orig_spacing, _ = res
        img_orig = process_xray(img_orig).astype(np.uint8)
        img_orig = LandmarkAnnotator.pad_img(img_orig, pad_img)

        h_orig, w_orig = img_orig.shape

        img = LandmarkAnnotator.resize_to_spacing(img_orig, orig_spacing, new_spacing)

        if return_orig:
            return img, orig_spacing, h_orig, w_orig, img_orig
        return img, orig_spacing, h_orig, w_orig

    @staticmethod
    def resize_to_spacing(img, spacing, new_spacing):
        if new_spacing is None:
            return img
        scale = spacing / new_spacing
        return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    def predict_img(self, img: tuple, h_orig=None, w_orig=None, rounded=True) -> np.ndarray:
        """Makes a prediction for the image. The image is expected to be a tuple of two images for the local search model as well.

        Args:
            img (tuple): _description_
            h_orig (int, optional): Original height. Defaults to None.
            w_orig (int, optional): Original width. Defaults to None.
            rounded (bool, optional): whether to round the landmarks or not. Defaults to True.

        Returns:
            np.ndarray: An array of landmarks
        """
        self.logger.log(logging.INFO, f'Running inference | {self.annotator_type}')
        # This wraps the image into the solt container
        # Implementation is stage-dependent
        dc = self.wrap_slt(img)
        # Running the augmentation
        roi_r, roi_l = self.trf(dc,  mean=self.mean_vector, std=self.std_vector)["images"]
        data_torched = torch.stack((roi_r, roi_l), dim=0)

        res = self.batch_inference(data_torched).squeeze()
        
        if self.annotator_type == 'global_search':
            res = self.handle_gs_out(res, h_orig, w_orig)
        else:
            res = self.handle_ls_out(res, h_orig, w_orig)
        if rounded:
            return np.round(res).astype(int)
        return res

    @staticmethod
    def handle_ls_out(res, h_orig, w_orig):
        res[:, :, 0] = w_orig * res[:, :, 0]
        res[:, :, 1] = h_orig * res[:, :, 1]
        return res

    @staticmethod
    def handle_gs_out(res, h_orig, w_orig):
        # right preds
        res[0, 0] = (w_orig // 2 + w_orig % 2) * res[0, 0]
        res[0, 1] = h_orig * res[0, 1]

        # left preds
        res[1, 0] = w_orig // 2 + w_orig // 2 * res[1, 0]
        res[1, 1] = h_orig * res[1, 1]

        return res

    @staticmethod
    def localize_left_right_rois(img, roi_size_pix, coords):
        s = roi_size_pix // 2
        roi_right = img[coords[0, 1] - s:coords[0, 1] + s,
                        coords[0, 0] - s:coords[0, 0] + s]

        roi_left = img[coords[1, 1] - s:coords[1, 1] + s,
                       coords[1, 0] - s:coords[1, 0] + s]

        return roi_right, roi_left

    def batch_inference(self, batch: torch.tensor):
        if batch.device != self.device:
            batch = batch.to(self.device)
        with torch.no_grad():
            res = self.net(batch)
        return res.to('cpu').numpy()

    def predict_local(self, img, center_coords, roi_size_px, orig_spacing):
        self.logger.log(logging.INFO, f'Running landmark prediction for image {img.shape}')
        if self.annotator_type != 'local_search':
            raise ValueError('This method can be called only for local search model')
        right_roi_orig, left_roi_orig = LandmarkAnnotator.localize_left_right_rois(img, roi_size_px, center_coords)

        left_roi_orig = left_roi_orig[:, ::-1]
        try:
            right_roi = LandmarkAnnotator.resize_to_spacing(right_roi_orig, orig_spacing, self.img_spacing)
        except cv2.error:
            right_roi = None

        try:
            left_roi = LandmarkAnnotator.resize_to_spacing(left_roi_orig, orig_spacing, self.img_spacing)
        except cv2.error:
            left_roi = None

        if left_roi is None and right_roi is None:
            return None, None, None
        elif left_roi is None:
            landmarks = self.predict_img((right_roi, right_roi.copy()),
                                         h_orig=roi_size_px,
                                         w_orig=roi_size_px)
            landmarks[1] = np.nan
            left_roi_orig = None
        elif right_roi is None:
            landmarks = self.predict_img((left_roi.copy(), left_roi),
                                         h_orig=roi_size_px,
                                         w_orig=roi_size_px)
            landmarks[0] = np.nan
            right_roi_orig = None
        else:
            landmarks = self.predict_img((right_roi, left_roi),
                                         h_orig=roi_size_px,
                                         w_orig=roi_size_px)

            left_roi_orig = left_roi_orig[:, ::-1]
            landmarks[1, :, 0] = roi_size_px - landmarks[1, :, 0]

        return landmarks, right_roi_orig, left_roi_orig