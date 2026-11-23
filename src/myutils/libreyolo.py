import math
from collections.abc import Iterable
from itertools import batched
from pathlib import Path

try:
    from libreyolo.models import LibreYOLO
except ImportError as e:
    raise ImportError(
        "Missing dependencies for myutils.libreyolo. Please install with extra flag: 'myutils[libreyolo]'"
    ) from e


class YOLORunner:
    def __init__(self, model_path: str = "models/LibreYOLO9c.pt"):
        self._model = LibreYOLO(model_path)
        self._name_to_cls = {str(name): int(id) for id, name in self._model.names.items()}

    def get_cls(self, name: str):
        return self._name_to_cls[name]

    def extract(self, paths: Iterable[Path], batch: int = 1):
        for paths_batch in batched(paths, batch):
            for path, result in zip(
                paths_batch,
                self._model(paths_batch, batch=batch, stream=True),
            ):
                if result.boxes is None:
                    continue

                for idx, (xyxy, cls) in enumerate(
                    zip(
                        (
                            (int(x1), int(y1), int(x2), int(y2))
                            for (x1, y1, x2, y2) in result.boxes.xyxy.tolist()
                        ),
                        (int(cls) for cls in result.boxes.cls.tolist()),
                    )
                ):
                    yield path, idx, xyxy, cls


def expand_bounding_box(
    xyxy: tuple[int, int, int, int],
    expansion_factors: float | tuple[float, float],
):
    if not isinstance(expansion_factors, tuple):
        expansion_factors = (expansion_factors, expansion_factors)

    dx = math.ceil((xyxy[2] - xyxy[0]) * expansion_factors[0] / 2)
    dy = math.ceil((xyxy[3] - xyxy[1]) * expansion_factors[1] / 2)

    return (
        max(xyxy[0] - dx, 0),
        max(xyxy[1] - dy, 0),
        xyxy[2] + dx,
        xyxy[3] + dy,
    )


__all__ = [
    "YOLORunner",
    "expand_bounding_box",
]
