import os.path as osp
import sys
import cv2
import argparse
__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)

from worker.worker3 import Worker3
def make_parser():
    parser = argparse.ArgumentParser("simple block work flow")
    parser.add_argument(
        "-vin", "--video_input",
        type=str, required=True,
        help="source to worker"
    )
    parser.add_argument(
        "-c", "--config",
        type=str, required=True,
        help="YAML format configuration to worker"
    )
    parser.add_argument(
        "-fl", "--frame_limit",
        type=int, default=-1,
        help="stop after frame amount"
    )
    parser.add_argument(
        "-vout", "--video_output",
        type=str, default=None,
        help="store worker visualize result"
    )
    parser.add_argument(
        "-so", "--stream_output",
        type=str, default=None,
        help="media server location for pushing stream with ffmpeg"
    )
    parser.add_argument(
        "--track_thresh",
        type=float, default=0.5,
        help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", 
        type=int, default=30,
        help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh", 
        type=float, default=0.8,
        help="matching threshold for tracking"
    )
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    track_parameter = {
        "track_thresh":args.track_thresh,
        "track_buffer":args.track_buffer,
        "match_thresh":args.match_thresh,
        "aspect_ratio_thresh":1.6,
        "min_box_area":10.0,
        "mot20":False,
    }
    worker = Worker3(
        args.video_input,
        args.config, track_parameter
    )

    if args.video_output is None:
        vwriter = None
    else:
        vwriter = cv2.VideoWriter(
            args.video_output,
            cv2.VideoWriter_fourcc(*'X264'),
            int(worker.cap.get(cv2.CAP_PROP_FPS)),
            (
                int(worker.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(worker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        )
    
    try:
        for fid, frame in worker:
            print(f"Now is => [ {fid / worker.fps} ]", end='\r')
            if vwriter is not None:
                vwriter.write(frame)
            if args.frame_limit > 1 and fid >= args.frame_limit:
                break
        print('\n')
    except KeyboardInterrupt:
        worker._endingWork()
        print("\nInterupt...Shutdown...ok...")
    except:
        raise
    finally:
        if vwriter is not None:
            vwriter.release()
