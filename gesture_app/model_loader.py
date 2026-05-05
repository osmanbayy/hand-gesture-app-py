from pathlib import Path
from urllib.request import urlretrieve


def ensure_hand_landmarker_model() -> Path:
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    if model_path.exists():
        return model_path

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    urlretrieve(model_url, model_path)
    return model_path
