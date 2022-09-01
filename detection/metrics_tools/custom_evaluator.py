import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'Object-Detection-Metrics', 'samples', 'sample_2'))

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator
from utils import *


class CustomEvaluator(Evaluator):
    def GetPascalVOCMetrics(self,
                            boundingboxes,
                            IOUThreshold=0.5,
                            method=MethodAveragePrecision.EveryPointInterpolation):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Get
        Args:
            boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
            (default value = 0.5);
            method (default = EveryPointInterpolation): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
            interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EveryPointInterpolation"  (ElevenPointInterpolation);
        Returns:
            A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Positive detections;
        """

        # CUSTOM INSERT
        gt_list = []  # list of ground truth bounding boxes
        tp_list = []  # list of true positive bounding boxes
        fp_list = []  # list of false positive bounding boxes

        ret = []  # list containing metrics (precision, recall, average precision) of each class
        # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
        groundTruths = []
        # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in boundingboxes.getBoundingBoxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                gt_list.append(bb)  # CUSTOM INSERT
                groundTruths.append([
                    bb.getImageName(),
                    bb.getClassId(), 1,
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            else:
                detections.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    bb.getConfidence(),
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        # Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c, use filename as key
            gts = {}
            npos = 0
            for g in groundTruths:
                if g[1] == c:
                    npos += 1
                    gts[g[0]] = gts.get(g[0], []) + [g]

            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            det = {key: np.zeros(len(gts[key])) for key in gts}

            # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
            # Loop through detections
            for d in range(len(dects)):
                # print('dect %s => %s' % (dects[d][0], dects[d][3],))
                # Find ground truth image
                gt = gts[dects[d][0]] if dects[d][0] in gts else []
                iouMax = sys.float_info.min
                for j in range(len(gt)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iouMax >= IOUThreshold:
                    if det[dects[d][0]][jmax] == 0:

                        # CUSTOM INSERT
                        image_name, class_id, confidence, bb_xyx2y2 = dects[d]
                        tp_list.append(BoundingBox(image_name,
                                                   class_id,
                                                   bb_xyx2y2[0], bb_xyx2y2[1],
                                                   bb_xyx2y2[2] - bb_xyx2y2[0],
                                                   bb_xyx2y2[3] - bb_xyx2y2[1],
                                                   classConfidence=confidence))

                        TP[d] = 1  # count as true positive
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                        # print("TP")
                    else:

                        # CUSTOM INSERT
                        image_name, class_id, confidence, bb_xyx2y2 = dects[d]
                        fp_list.append(BoundingBox(image_name,
                                                   class_id,
                                                   bb_xyx2y2[0], bb_xyx2y2[1],
                                                   bb_xyx2y2[2] - bb_xyx2y2[0],
                                                   bb_xyx2y2[3] - bb_xyx2y2[1],
                                                   classConfidence=confidence))

                        FP[d] = 1  # count as false positive
                        # print("FP")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:

                    # CUSTOM INSERT
                    image_name, class_id, confidence, bb_xyx2y2 = dects[d]
                    fp_list.append(BoundingBox(image_name,
                                               class_id,
                                               bb_xyx2y2[0], bb_xyx2y2[1],
                                               bb_xyx2y2[2] - bb_xyx2y2[0],
                                               bb_xyx2y2[3] - bb_xyx2y2[1],
                                               classConfidence=confidence))

                    FP[d] = 1  # count as false positive
                    # print("FP")
            # compute precision, recall and average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EveryPointInterpolation:
                [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),

                # CUSTOM INSERT
                'gt_list': gt_list,
                'tp_list': tp_list,
                'fp_list': fp_list,
            }
            ret.append(r)
        return ret
