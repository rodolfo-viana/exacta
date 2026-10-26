"""Differentiable losses (PyTorch modules)."""

from exacta.losses.segmentation import (
    DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss, FocalTverskyLoss, GeneralizedDiceCELoss, GeneralizedDiceLoss, TverskyLoss,
)
from exacta.losses.deep_supervision import DeepSupervisionLoss
from exacta.losses.ranking import TopKCELoss, DiceTopKCELoss, LovaszSoftmaxLoss, LovaszHingeLoss
from exacta.losses.geometry import BoundaryLoss, HausdorffDTLoss, distance_maps
from exacta.losses.topology import SoftClDiceLoss, soft_skeletonize
from exacta.losses.wasserstein import GeneralizedWassersteinDiceLoss

__all__ = ["GeneralizedWassersteinDiceLoss", "DeepSupervisionLoss", "DiceCELoss", "DiceFocalLoss", "DiceLoss", "FocalLoss",
           "GeneralizedDiceCELoss", "GeneralizedDiceLoss", "TverskyLoss", "FocalTverskyLoss",
           "TopKCELoss", "DiceTopKCELoss", "LovaszSoftmaxLoss", "LovaszHingeLoss", "BoundaryLoss", "HausdorffDTLoss",
           "distance_maps", "SoftClDiceLoss", "soft_skeletonize"]

from exacta.losses import classification, regression

__all__ += ["classification", "regression"]
