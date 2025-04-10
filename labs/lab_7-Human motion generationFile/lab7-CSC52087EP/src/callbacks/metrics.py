from typing import Any, Dict

from torchtyping import TensorType

from src.metrics.frechet import FrechetDistance
from src.metrics.similarity import SimilarityScore
from src.metrics.motion_text import MotionTextMetric
from src.metrics.prdc import ManifoldMetrics


# ------------------------------------------------------------------------------------- #

num_samples, num_frames = None, None
num_vertices, num_faces, num_feats = None, None, None
Traj = TensorType["num_samples", "num_frames", 4, 4]
Feat = TensorType["num_samples", "num_frames", "num_feats"]
Mask = TensorType["num_samples", "num_frames"]
Proj = TensorType["num_samples", "num_frames", "height", "width"]
Verts = TensorType["num_samples", "num_frames", "num_vertices", 3]
Faces = TensorType["num_samples", "num_frames", "num_faces", 3]

# ------------------------------------------------------------------------------------- #


class MetricCallback:
    def __init__(
        self,
        num_frames: int,
        device: str,
        num_train_samples: int = 1,
        num_val_samples: int = 1,
        num_test_samples: int = 1,
    ):
        self.num_frames = num_frames

        self.tmr_fd = {
            "train": FrechetDistance(num_features=256),
            "val": FrechetDistance(num_features=256),
            "test": FrechetDistance(num_features=256),
        }
        self.tmr_prdc = {
            "train": ManifoldMetrics(distance="euclidean"),
            "val": ManifoldMetrics(distance="euclidean"),
            "test": ManifoldMetrics(distance="euclidean"),
        }
        self.tmr_motiontext = {
            "train": MotionTextMetric(top_k=[1, 2, 3]),
            "val": MotionTextMetric(top_k=[1, 2, 3]),
            "test": MotionTextMetric(top_k=[1, 2, 3]),
        }
        self.tmr_score = {
            "train": SimilarityScore(),
            "val": SimilarityScore(),
            "test": SimilarityScore(),
        }

        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str) -> Feat:
        for stage in ["train", "val", "test"]:
            self.tmr_motiontext[stage].to(device)
            self.tmr_fd[stage].to(device)
            self.tmr_prdc[stage].to(device)

    # --------------------------------------------------------------------------------- #

    def update_tmr_metrics(
        self,
        stage: str,
        pred: Feat,
        ref: Feat,
        text: Feat,
        batch_indices: TensorType["num_samples"],
    ):
        self.tmr_score[stage].update(pred, text)
        self.tmr_motiontext[stage].update(pred, text)

        self.tmr_prdc[stage].update(pred, ref)
        self.tmr_fd[stage].update(pred, ref)

    def compute_tmr_metrics(self, stage: str) -> Dict[str, Any]:
        tmr_score = self.tmr_score[stage].compute()
        self.tmr_score[stage].reset()

        fid = self.tmr_fd[stage].compute()
        self.tmr_fd[stage].reset()

        tmr_motiontext = self.tmr_motiontext[stage].compute()
        self.tmr_motiontext[stage].reset()

        tmr_p, tmr_r, tmr_d, tmr_c = self.tmr_prdc[stage].compute()
        self.tmr_prdc[stage].reset()

        return {
            "tmr/fid": fid,
            "tmr/tmr_score": tmr_score,
            "tmr/precision": tmr_p,
            "tmr/recall": tmr_r,
            "tmr/density": tmr_d,
            "tmr/coverage": tmr_c,
            **{f"tmr/{k}": v for k, v in tmr_motiontext.items()},
        }
