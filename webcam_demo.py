import os
import argparse
import time
import cv2
import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.core.evaluation import decode_preds
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser(description='Real-time webcam demo')

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='experiments/wflw/face_alignment_wflw_hrnet_w18.yaml')
    parser.add_argument('--model-file', help='model parameters', type=str,
                        default='HR18-WFLW.pth')

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    # load model
    state_dict = torch.load(args.model_file)
    model.load_state_dict(state_dict)
    model.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        start = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, [256,256])
        input = np.copy(frame).astype(np.float32)
        input = (input / 255.0 - mean) / std
        input = torch.from_numpy(input).permute(2,0,1).unsqueeze(0)

        output = model(input)
        score_map = output.data.cpu()
        preds = decode_preds(score_map, torch.tensor([[128.0,128.0]]), torch.tensor([1.28]), [64, 64])

        for x,y in preds[0]:
            x, y = int(x), int(y)
            cv2.circle(frame, (x,y), 1, (0, 0, 255), 2)
        
        fps = 1 / (time.time() - start)
        cv2.putText(frame, text=f'FPS: {round(fps,2)}', org=[10, 30], fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
            fontScale=0.7, color=(0,200,0), thickness=2)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()