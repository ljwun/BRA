import os
import os.path as osp
import sys
__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)
sys.path.append(osp.join(__proot__,  "third_party", "YOLOX"))

import argparse
import yaml
from pipe_block import FrameCenter
from detect import Detector
import cv2
import ujson as json

def make_parser():
    parser = argparse.ArgumentParser("Generate self dataset prototype")
    parser.add_argument(
        "-ssize", "--step_size",
        type=int, default=1,
        help="step group just retrieves first frame, rest will be ignored"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=1,
        help="number of frame will be processed at once"
    )
    parser.add_argument(
        "--detect_thresh",
        type=float, default=0.5,
        help="detection confidence threshold"
    )
    parser.add_argument(
        "-info", "--dataset_info",
        type=str, required=True,
        help="detection confidence threshold"
    )
    parser.add_argument(
        "-t", "--target_destination",
        type=str, required=True,
        help="detection confidence threshold"
    )
    parser.add_argument(
        "--yolox_type",
        type=str, default='m',
        help="choose yolox type for detection person"
    )
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    with open(args.dataset_info, 'r') as stream:
        dataset_info = yaml.safe_load(stream)
    PDetector = Detector(args.yolox_type, 0, True, True)
    for type in dataset_info['video'].keys():
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        ann_folder = osp.join(osp.normpath(args.target_destination), 'annotations')
        if not osp.exists(ann_folder):
            os.makedirs(ann_folder)
        out_path = osp.join(ann_folder, f'{type}.json')
        img_folder_parent = osp.join(args.target_destination, type)
        if not osp.exists(img_folder_parent):
            os.makedirs(img_folder_parent)
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        for scene in dataset_info['video'][type]:
            img_folder = osp.join(img_folder_parent, f'{dataset_info["name"]}-{scene["tag"]}')
            if not osp.exists(img_folder):
                os.makedirs(img_folder)
            if 'step_size' not in scene['property'] or scene['property']['step_size'] is None:
                scene['property']['step_size'] = args.step_size
            if 'detect_thresh' not in scene['property'] or scene['property']['detect_thresh'] is None:
                scene['property']['detect_thresh'] = args.detect_thresh
            FCenter = FrameCenter(
                osp.join(osp.normpath(dataset_info['video_folder']), osp.normpath(scene['path'])), 
                max_batch=args.batch_size, 
                frame_step=scene['property']['step_size']
            )
            print(f'Starting process {type}/{scene["tag"]} folder')
            while True:
                FCenter.Load()
                frames, length, fids = FCenter.Allocate()
                if FCenter.Finished and length == 0:
                    break
                p_outputs, p_infos = PDetector.detect(frames)
                for p_output, p_info in zip(p_outputs, p_infos):
                    if len(p_output) < 1:
                        continue
                    else:
                        image_cnt += 1
                    img_path = osp.join(img_folder, f'{image_cnt:06d}.jpg')
                    cv2.imwrite(img_path, p_info['raw'])
                    image_info = {
                        'file_name': osp.join(dataset_info['name'], type, f'{dataset_info["name"]}-{scene["tag"]}', f'{image_cnt:06d}.jpg'), 
                        'id': image_cnt,
                        'height': p_info['height'], 
                        'width': p_info['width']
                    }
                    out['images'].append(image_info)

                    p_output = p_output.cpu().numpy()
                    scores = p_output[:, 4] * p_output[:, 5]
                    bboxes = p_output[:, :4]
                    bboxes = bboxes / p_info['ratio']
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
                    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
                    for bbox, score in zip(bboxes, scores):
                        if score < scene['property']['detect_thresh']:
                            continue
                        bbox = bbox.tolist()
                        bbox = [round(v) for v in bbox]
                        ann_cnt += 1
                        ann = {
                            'id': ann_cnt,
                            'category_id': 1,
                            'image_id': image_cnt,
                            'track_id': -1,
                            'bbox': bbox,
                            'area': bbox[2] * bbox[3],
                            'iscrowd': 0
                        }
                        out['annotations'].append(ann)
            FCenter.Exit()
        print(f"loaded {type} for {len(out['images'])} images and {len(out['annotations'])} samples")
        json.dump(out, open(out_path, 'w'))
