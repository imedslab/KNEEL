import numpy as np
import logging
from .annotator import LandmarkAnnotator
from kneel.model.hf import load_models

class KneeAnnotatorPipeline(object):
    def __init__(self, hf_commit, cache, hf_token, device, jit_trace=True):
        self.logger = logging.getLogger(f'kneel-backend:pipeline')
        models, (mean, std) = load_models(hf_token, hf_commit, cache)
        self.logger.log(logging.INFO, 'Initializing the global searcher (ROI localizer)')
        
        self.global_searcher = LandmarkAnnotator(models["global_search"], mean, std, "global_search",
                                                 device=device,
                                                 jit_trace=jit_trace,
                                                 logger=logging.getLogger(f'kneel-backend:global_search'))

        self.logger.log(logging.INFO, 'Initializing the local searcher (landmark localizer)')
        self.local_searcher = LandmarkAnnotator(models["local_search"], mean, std, "local_search", 
                                                device=device,
                                                jit_trace=jit_trace,
                                                logger=logging.getLogger(f'kneel-backend:local_search'))

    def predict(self, img_name, roi_size_mm=140, pad=300, refine=True):
        self.logger.log(logging.INFO, f'Loading the image with a new spacing of {self.global_searcher.img_spacing} mm.')
        res = self.global_searcher.read_dicom(img_name,
                                              new_spacing=self.global_searcher.img_spacing,
                                              return_orig=True)
        if len(res) > 0:
            img, orig_spacing, h_orig, w_orig, img_orig = res
        else:
            return None

        # First pass of knee joint center estimation
        self.logger.log(logging.INFO, 'Predicting knee joint centers')
        roi_size_px = int(roi_size_mm * 1. / orig_spacing)
        global_coords = self.global_searcher.predict_img(img, h_orig, w_orig)

        img_orig = LandmarkAnnotator.pad_img(img_orig, pad if pad != 0 else None)
        global_coords += pad
        self.logger.log(logging.INFO, 'Predicting knee landmarks')
        landmarks, _, _ = self.local_searcher.predict_local(img_orig, global_coords, roi_size_px, orig_spacing)

        if refine:
            # refinement
            self.logger.log(logging.INFO, 'Refining the predictions via second pass through the landmark localizer.')
            centers_d = np.array([roi_size_px // 2, roi_size_px // 2]) - landmarks[:, 4]
            global_coords -= centers_d
            # prediction for refined centers
            landmarks, _, _ = self.local_searcher.predict_local(img_orig, global_coords, roi_size_px, orig_spacing)
        landmarks -= pad
        landmarks[0, :, :] += global_coords[0, :] - roi_size_px // 2
        landmarks[1, :, :] += global_coords[1, :] - roi_size_px // 2
        landmarks = np.expand_dims(landmarks, 0)
        return landmarks
