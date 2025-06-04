import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


class YoloModel:
    def __init__(
        self,
        model_path: str = "best.pt",
        conf_threshold: float | None = 0.1,
        ensure_onnx: bool = True,
        verbose: bool = False,
    ):
        if model_path.endswith(".pt") and ensure_onnx:
            self.convert_onnx(model_path)
            model_path = model_path.replace(".pt", ".onnx")

        self.model = YOLO(model=model_path, task="detect")

        if model_path.endswith(".pt") and not ensure_onnx:
            self.model = self.model.to("cpu")

        self.conf_threshold = conf_threshold
        self.verbose = verbose

    def convert_onnx(self, model_path: str):
        model = YOLO(model_path)

        # Export to ONNX format
        model.export(
            format="onnx",
            dynamic=True,  # Dynamic input size
            simplify=True,
            device="cpu",
        )

        print(
            "Model successfully converted to ONNX and saved at ",
            model_path.replace(".pt", ".onnx"),
        )

    def infer(self, image: np.ndarray) -> list[dict]:
        results = self.model.predict(
            image,
            verbose=self.verbose,
            conf=self.conf_threshold,
        )[0]
        detections = []
        for box in results.boxes:
            if box.conf >= self.conf_threshold:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),  # bounding box [x1, y1, x2, y2]
                        "conf": float(box.conf),
                        "cls": int(box.cls),
                    }
                )

        return detections
