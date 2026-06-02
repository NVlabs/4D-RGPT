import os
import cv2
import base64
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from moviepy import VideoFileClip, ImageSequenceClip
from typing import List, Tuple, Any, Optional


def mp42gif(
    video_path: str,
    out_path: str,
    num_frames: int = 10
) -> None:
    clip = VideoFileClip(video_path)
    # clip = clip.resize(height=300)

    timestamps = np.linspace(0, clip.duration, num=num_frames, endpoint=False)
    frames = [clip.get_frame(t) for t in timestamps]
    gif_clip = ImageSequenceClip(frames, fps=5)  # 2 fps = 5 seconds total
    gif_clip.write_gif(out_path, logger=None)
    gif_clip.close()

def images2gif(
    image_paths: List[str],
    out_path: str,
    fps: int = 5
) -> None:
    gif_clip = ImageSequenceClip(image_paths, fps=fps)  # 2 fps = 5 seconds total
    gif_clip.write_gif(out_path)


def get_first_frame(video_path: str, format: str = 'BGR') -> np.ndarray:
    r'''
    Return cv2 image in BGR by default
    '''
    cap = cv2.VideoCapture(video_path)
    _, cv2_image = cap.read()
    cap.release()
    if format == 'BGR':
        return cv2_image
    elif format == 'RGB':
        return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unknown format {format}")


def print_stats(
    dataset_name: str,
    all_tasks: list[str] | dict[str, list[str]],
    test_num_videos: int = 0,
    test_num_qas: int = 0,
    train_num_videos: int = 0,
    train_num_qas: int = 0,
    fps: Optional[float] = None,
) -> None:
    print(f"Dataset Overview of {dataset_name}")
    print("-"*20)
    print(f"- Number of Tasks                   : {len(all_tasks)}")
    if fps is not None:
        if isinstance(fps, list):
            print(f"- Video FPS                         : {[f'{f:.2f}' for f in fps]}")
        else:
            print(f"- Video FPS                         : {fps:.2f}")

    for task in all_tasks:
        print(f"    - {task}")
        if isinstance(all_tasks, dict):
            for subtask in all_tasks[task]:
                print(f"        - {subtask}")
    if train_num_videos > 0:
        print(f"- Number of Training Visual Inputs  : {train_num_videos:,}")
    if train_num_qas > 0:
        print(f"- Number of Training QA Pairs       : {train_num_qas:,}")
    if test_num_videos > 0:
        print(f"- Number of Testing Visual Inputs   : {test_num_videos:,}")
    if test_num_qas > 0:
        print(f"- Number of Testing QA Pairs        : {test_num_qas:,}")
    print("-"*20)

def extract_zip(in_path: str, out_dir: str, desc: str = 'Extracting ', verbose: bool = True) -> None:
    import zipfile
    if verbose:
        with zipfile.ZipFile(in_path, 'r') as zf:
            for member in tqdm(zf.infolist(), desc=desc):
                try:
                    zf.extract(member, out_dir)
                except zipfile.error as e:
                    pass
    else:
        with zipfile.ZipFile(in_path, 'r') as zf:
            zf.extractall(out_dir)


def single_mask_to_rle(mask):
    import pycocotools.mask as mask_util
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def get_bbox_from_davis_mask(label_map: np.ndarray, label: int) -> Tuple[int, int, int, int]:
    """
    Given a HxW segmentation mask, return a (xmin, ymin, xmax, ymax).
    """
    ys, xs = np.where(label_map == label)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return (xmin, ymin, xmax, ymax)

def mask_to_bbox(mask: np.ndarray, format='xyxy') -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return (xmin, ymin, xmax, ymax)

def set_of_marks(
    cv2_image: np.ndarray,
    labels : List[str],
    input_boxes: np.ndarray,
    masks: np.ndarray | None = None
) -> np.ndarray:
    """
    Visualize image with supervision useful API
    """
    import supervision as sv
    class_ids = np.array(list(range(len(labels))))

    annotated_frame = cv2_image.copy()

    if masks is None:
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

    else:
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

def download_with_aria2c(
    url: str,
    output_dir: str = "./",
    connections: int = 16,
    interval: int = 0,
    check_certificate: bool = True,
) -> None:
    cmd = [
        "aria2c",
        "--dir", output_dir,
        "-x", str(connections),  # number of connections per server
        f"--summary-interval={interval}",
        "--check-integrity=true",
        f"--check-certificate={'true' if check_certificate else 'false'}",
        "--continue",  # resume if file exists
        url
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")

def download_huggingface_model(
    model_name: str = "Efficient-Large-Model/NVILA-Lite-8B-hf-preview"
) -> None:
    from huggingface_hub import snapshot_download
    hf_home_dir = os.environ.get("HF_HOME", './cache')
    # target_dir = os.path.join(hf_home_dir, 'cache', ('models/' + model_name).replace('/', '--'))
    # print(target_dir)

    # if os.path.exists(target_dir):
    #     print(f"Model {model_name} already exists at {target_dir}")
    # else:
    #     print("Downloading model...")
    snapshot_path = snapshot_download(
        repo_id=model_name,
        cache_dir=os.path.join(hf_home_dir, 'hub_cache'),
    )
    print(f"Model {model_name} saved at: {snapshot_path}")

def extract_video_clip(input_path: str, output_path: str, start_time: str, end_time: str) -> None:
    """
    Extracts a frame from a video file at a specific timestamp using `ffmpeg`.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output image file.
        start_time (str): Timestamp of the start frame to extract. Format: 'hh:mm:ss.ms'
        end_time (str): Timestamp of the end frame to extract. Format: 'hh:mm:ss.ms'
    """
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "warning",
        "-i", input_path,
        "-ss", start_time,
        "-to", end_time,
        "-y"
    ]
    result = subprocess.run(ffmpeg_command + ["-c", "copy"] + [output_path], check=True)
    if result.returncode != 0:
        result = subprocess.run(ffmpeg_command + [output_path], check=True)

def extract_video_frame(input_path: str, output_path: str, frame_time: str) -> None:
    """
    Extracts a frame from a video file at a specific timestamp using `ffmpeg`.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output image file.
        frame_time (str): Timestamp of the frame to extract. Format: 'hh:mm:ss.ms'
    """
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "warning",
        "-i", input_path,
        "-ss", frame_time,     # Seek to timestamp
        "-frames:v", "1",      # Only extract one frame
        "-y",
    ]

    # Run the ffmpeg command to extract a frame from the input video file.
    result = subprocess.run(ffmpeg_command + ["-c", "copy"] + [output_path], check=True)
    if result.returncode != 0:
        result = subprocess.run(ffmpeg_command + [output_path], check=True)



def seconds_to_hhmmss_ms(sec):
    td = timedelta(seconds=sec)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((sec - int(sec)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# def get_fps(video_path: str) -> int:
#     clip = VideoFileClip(video_path)
#     return int(clip.fps)


def get_video_height_width(video_path: str) -> Tuple[int, int]:
    clip = VideoFileClip(video_path)

    width, height = clip.size
    return height, width

def get_video_length(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0

def sample_video_frames(video_path: str, n_frames: int = 8) -> tuple[list[str], float]:
    """
    Sample N frames uniformly from a video, encode them in base64,
    and return (base64_frames, frames_per_sec_sampled).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    # Uniformly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frame_paths = []

    # prepare output folder: x/y/a/
    out_dir = video_path.with_suffix('')  # removes .mp4
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in frame_indices:
        frame_path = out_dir / f"{i:04d}.png"
        if not frame_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path.as_posix())

    cap.release()

    # Average sampling rate (frames sampled per second)
    frames_per_sec_sampled = n_frames / duration if duration > 0 else 0

    return frame_paths, frames_per_sec_sampled