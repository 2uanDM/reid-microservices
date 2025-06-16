import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


class YoloModel:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float | None = 0.25,
        ensure_onnx: bool = True,
        ensure_openvino: bool = True,
        verbose: bool = False,
    ):
        if ensure_onnx and ensure_openvino:
            raise ValueError(
                "ensure_onnx and ensure_openvino cannot be both True at the same time"
            )

        if model_path.endswith(".pt") and ensure_openvino:
            self.convert_open_vino(model_path)
            model_path = "weights/yolo11n_openvino_model"

        if model_path.endswith(".pt") and ensure_onnx:
            self.convert_onnx(model_path)
            model_path = model_path.replace(".pt", ".onnx")

        self.model = YOLO(model=model_path, task="detect")

        if model_path.endswith(".pt") and not ensure_onnx and not ensure_openvino:
            self.model = self.model.to("cpu")

        self.conf_threshold = conf_threshold
        self.verbose = verbose

    def convert_onnx(self, model_path: str):
        model = YOLO(model_path)

        model.export(
            format="onnx",
            dynamic=True,
            simplify=True,
        )

        print(
            "Model successfully converted to ONNX and saved at ",
            model_path.replace(".pt", ".onnx"),
        )

    def convert_open_vino(self, model_path: str):
        model = YOLO(model_path)
        model.export(
            format="openvino",
            dynamic=True,
            simplify=True,
        )

        print(
            "Model successfully converted to OpenVINO and saved at ",
            model_path.replace(".pt", ".openvino"),
        )

    def infer(self, image: np.ndarray) -> list[dict]:
        results = self.model(
            image,
            verbose=self.verbose,
            conf=self.conf_threshold,
            classes=[0],
        )[0]
        detections = []
        for box in results.boxes:
            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),  # bounding box [x1, y1, x2, y2]
                    "confidence": float(box.conf),
                    "class_id": int(box.cls),
                }
            )

        return detections
