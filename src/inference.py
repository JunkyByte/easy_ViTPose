import abc
import argparse
from collections import deque
import json
import os

from PIL import Image
import cv2
import numpy as np
import torch

from vit_models.model import ViTPose
from vit_utils.top_down_eval import keypoints_from_heatmaps
from vit_utils.visualization import draw_points_and_skeleton, joints_dict
from vit_utils.inference import pad_image, VideoReader
from sort import Sort

try:  # Add bools -> error stack
    import pycuda.driver as cuda  # noqa: [F401]
    import pycuda.autoinit  # noqa: [F401]
    import utils_engine as engine_utils
    import tensorrt as trt
    has_trt = True
except ModuleNotFoundError:
    pass

try:
    import onnxruntime
    has_onnx = True
except ModuleNotFoundError:
    pass

__all__ = ['VitInference']


# TODO: Where should this go?
def draw_bboxes(image, bounding_boxes, boxes_id, scores):
    image_with_boxes = image.copy()

    for bbox, bbox_id, score in zip(bounding_boxes, boxes_id, scores):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (128, 128, 0), 2)

        label = f'#{bbox_id}: {score:.2f}'

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 5 if y1 > 20 else y1 + 20

        # Draw a filled rectangle as the background for the label
        cv2.rectangle(image_with_boxes, (x1, label_y - label_height - 5),
                      (x1 + label_width, label_y + 5), (128, 128, 0), cv2.FILLED)
        cv2.putText(image_with_boxes, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image_with_boxes


class VitInference:
    def __init__(self, model, yolo_name, model_name=None, yolo_size=320,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 is_video=False):
        self.device = torch.device(device)
        self.yolo = torch.hub.load("ultralytics/yolov5", "custom", yolo_name)
        self.yolo.classes = [0]
        self.yolo_size = yolo_size
        self.tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3) if is_video else None  # TODO: Params
        self.is_video = is_video
        self.waitKey = 1 if is_video else 0

        # Use extension to decide which kind of model has been loaded
        use_onnx = model.endswith('.onnx')
        use_trt = model.endswith('.engine')

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # onnx / trt models do not require model_cfg specification, but we need img size
        # TODO: These can be replaced, during inference they are almost useless, only
        # needed for img size and torch version of the model.
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify the model_name if not using onnx / trt'
            model_name = 's'

        if model_name == 's':
            from configs.ViTPose_small_coco_256x192 import model as model_cfg
            from configs.ViTPose_small_coco_256x192 import data_cfg
        elif model_name == 'b':
            from configs.ViTPose_base_coco_256x192 import model as model_cfg
            from configs.ViTPose_base_coco_256x192 import data_cfg
        elif model_name == 'l':
            from configs.ViTPose_large_coco_256x192 import model as model_cfg
            from configs.ViTPose_large_coco_256x192 import data_cfg
        elif model_name == 'h':
            from configs.ViTPose_huge_coco_256x192 import model as model_cfg
            from configs.ViTPose_huge_coco_256x192 import data_cfg

        self.target_size = data_cfg['image_size']
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(model,
                                                             providers=['CUDAExecutionProvider',
                                                                        'CPUExecutionProvider'])
            inf_fn = self._inference_onnx
        elif use_trt:
            logger = trt.Logger(trt.Logger.ERROR)
            trt_runtime = trt.Runtime(logger)
            trt_engine = engine_utils.load_engine(trt_runtime, model)

            # This allocates memory for network inputs/outputs on both CPU and GPU
            self._inputs, self._outputs, self._bindings, self._stream = \
                engine_utils.allocate_buffers(trt_engine)
            # Execution context is needed for inference
            self._context = trt_engine.create_execution_context()
            inf_fn = self._inference_trt
        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            ckpt = torch.load(model, map_location='cpu')
            if 'state_dict' in ckpt:
                self._vit_pose.load_state_dict(ckpt['state_dict'])
            else:
                self._vit_pose.load_state_dict(ckpt)
            self._vit_pose.to(device)
            inf_fn = self._inference_torch

        # Override inference with selected engine
        self._inference = inf_fn

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array([[org_w // 2,
                                                                 org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    @abc.abstractmethod
    def _inference(img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inference(self, img: np.ndarray, show=False, show_yolo=False) -> np.ndarray:
        # First use YOLOv5 for detection
        results = self.yolo(img, size=self.yolo_size)
        # TODO: Confidence finetuning, is it done by tracker? should do it only on non tracker?
        res_pd = np.array([r[:5] for r in results.pandas().xyxy[0].to_numpy() if r[4] > 0.4])

        frame_keypoints = {}
        if self.tracker is not None:
            res_pd = self.tracker.update(res_pd)
        bboxes = res_pd[:, :4].round().astype(int)
        scores = res_pd[:, 4].tolist()
        ids = res_pd[:, 5].astype(int).tolist()

        for bbox, id in zip(bboxes, ids):
            # TODO: Slightly bigger bbox
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-5, 5], 0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-5, 5], 0, img.shape[0])

            # Crop image and pad to 3/4 aspect ratio
            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            keypoints = self._inference(img_inf)[0]
            # Transform keypoints to original image
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints[id] = keypoints

        if show:
            if show_yolo:
                # TODO: Add tracker rendering
                # img = np.array(results.render())[0]
                img = draw_bboxes(img, bboxes, ids, scores)

            img = np.array(img)[:, :, ::-1]  # RGB to BGR for cv2 modules
            for idx, k in frame_keypoints.items():
                img = draw_points_and_skeleton(img.copy(), k,
                                               joints_dict()['coco']['skeleton'],
                                               person_index=idx,
                                               points_color_palette='gist_rainbow',
                                               skeleton_color_palette='jet',
                                               points_palette_samples=10,
                                               confidence_threshold=0.4)

            cv2.imshow('preview', img)
            cv2.waitKey(self.waitKey)
        return frame_keypoints, results, res_pd  # TODO: Decide output! adapt in colab and here

    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:

        # Prepare input data
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255
        img_input = torch.from_numpy(img_input).to(self.device)

        # Feed to model
        heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:

        # Prepare input data
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255

        # Feed to model
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_trt(self, img: np.ndarray) -> np.ndarray:

        # Prepare input data
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255

        # Copy the data to appropriate memory
        np.copyto(self._inputs[0].host, img_input.ravel())

        heatmaps = engine_utils.do_inference(context=self._context,
                                             bindings=self._bindings,
                                             inputs=self._inputs,
                                             outputs=self._outputs,
                                             stream=self._stream)[0]

        # Reshape to output size
        heatmaps = heatmaps.reshape(1, 25, img_input.shape[2] // 4, img_input.shape[3] // 4)
        return self.postprocess(heatmaps, org_w, org_h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='examples/sample.jpg',
                        help='image or video path')
    parser.add_argument('--output-path', type=str, default='', help='output path')
    parser.add_argument('--model', type=str, required=True, help='ckpt path')
    parser.add_argument('--model-name', type=str, required=False,
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo-size', type=int, required=False, default=320,
                        help='YOLOv5 image size during inference')
    parser.add_argument('--yolo-nano', default=False, action='store_true',
                        help='Whether to use (the very fast) yolo nano (instead of small)')
    parser.add_argument('--show', default=False, action='store_true',
                        help='preview result')
    parser.add_argument('--show-yolo', default=False, action='store_true',
                        help='preview yolo result')
    parser.add_argument('--save-img', default=False, action='store_true',
                        help='save image result')
    parser.add_argument('--save-json', default=False, action='store_true',
                        help='save json result')
    args = parser.parse_args()

    # Load Yolo
    model_name = 'yolov5n' if args.yolo_nano else 'yolov5s'
    yolo_model = model_name + ('.onnx' if has_onnx else '.pt')

    input_path = args.input
    ext = input_path[input_path.rfind('.'):]

    # Load the image / video reader
    try:  # Check if is webcam
        int(input_path)
        is_video = True
    except ValueError:
        assert os.path.isfile(input_path), 'The input file does not exist'
        is_video = input_path[input_path.rfind('.') + 1:] in ['mp4']

    wait = 0
    if is_video:
        reader = VideoReader(input_path)
        wait = 15
        if args.save_img:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            cap.release()
            assert ret
            assert fps > 0
            output_size = frame.shape[:2][::-1]
            save_name = os.path.basename(input_path).replace(ext, f"_result{ext}")
            out_writer = cv2.VideoWriter(os.path.join(args.output_path, save_name),
                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                         fps, output_size)
    else:
        reader = [np.array(Image.open(input_path))]

    # Initialize model
    model = VitInference(args.model, yolo_model, args.model_name, args.yolo_size, is_video=is_video)
    print(f">>> Model loaded: {args.model}")

    print(f'Running inference on {input_path}')
    keypoints = []
    fps = deque([], maxlen=30)
    for ith, img in enumerate(reader):

        # Run inference
        frame_keypoints, yolo_res, bboxes = model.inference(img, show=args.show, show_yolo=args.show_yolo)
        keypoints.append(frame_keypoints)  # TODO: Here mantain PID with tracking

        # Draw the poses and save the output img
        if args.save_img:
            if args.show_yolo:
                img = np.array(yolo_res.render())[0]
                # img = draw_bboxes(img, bboxes)  # TODO

            img = np.array(img)[:, :, ::-1]  # RGB to BGR for cv2 modules
            for k in frame_keypoints:
                img = draw_points_and_skeleton(img.copy(), k,
                                               joints_dict()['coco']['skeleton'],
                                               person_index=0,  # TODO: Add pid
                                               points_color_palette='gist_rainbow',
                                               skeleton_color_palette='jet',
                                               points_palette_samples=10,
                                               confidence_threshold=0.4)

            if is_video:
                out_writer.write(img)
            else:
                save_name = os.path.basename(input_path).replace(ext, f"_result{ext}")
                cv2.imwrite(os.path.join(args.output_path, save_name), img)

    if args.save_json:
        print('>>> Saving output json')
        save_name = os.path.basename(input_path).replace(ext, "_result.json")
        with open(os.path.join(args.output_path, save_name), 'w') as f:
            out = {'keypoints': keypoints}
            out['skeleton'] = joints_dict()['coco']['keypoints']
            json.dump(out, f)

    if is_video and args.save_img:
        out_writer.release()
    cv2.destroyAllWindows()
