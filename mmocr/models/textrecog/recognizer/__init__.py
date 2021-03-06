# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseRecognizer
from .crnn import CRNNNet
from .encode_decode_recognizer import EncodeDecodeRecognizer
from .nrtr import NRTR
from .robust_scanner import RobustScanner
from .sar import SARNet
from .satrn import SATRN
from .seg_recognizer import SegRecognizer
from .master import MASTER
__all__ = [
    'BaseRecognizer', 'EncodeDecodeRecognizer', 'CRNNNet', 'SARNet', 'NRTR',
    'SegRecognizer', 'RobustScanner', 'SATRN', 'MASTER'
]
