"""Extract uniformly sampled frames from project videos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
from tqdm import tqdm

# Edit this section before running the script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "template"
PROJECT_NAME = "example"
FRAMES_PER_SECOND = 3.0
IMAGE_EXTENSION = ".jpg"
OVERWRITE_EXISTING = False
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def video_directory() -> Path:
    """Return the configured project video directory."""
    return PROJECT_ROOT / "datasets" / DATASET_NAME / "videos" / PROJECT_NAME


def output_directory() -> Path:
    """Return the dataset's unlabeled image directory."""
    return PROJECT_ROOT / "datasets" / DATASET_NAME / "images" / "unlabeled"


def iter_videos(directory: Path) -> Iterator[Path]:
    """Yield supported videos in deterministic order."""
    yield from sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def extract_video(video_path: Path, output_path: Path, next_index: int) -> int:
    """Extract frames from one video and return the next available index."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"无法打开视频，跳过: {video_path}")
        return next_index

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or FRAMES_PER_SECOND <= 0:
        print(f"视频帧率无效，跳过: {video_path}")
        capture.release()
        return next_index

    frame_interval = max(1, round(fps / FRAMES_PER_SECOND))
    frame_index = 0
    saved_count = 0
    progress = tqdm(total=total_frames, desc=video_path.name, unit="frame")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                image_path = output_path / f"{next_index:06d}{IMAGE_EXTENSION}"
                if OVERWRITE_EXISTING or not image_path.exists():
                    if not cv2.imwrite(str(image_path), frame):
                        raise RuntimeError(f"无法写入图片: {image_path}")
                    next_index += 1
                    saved_count += 1
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()

    print(f"{video_path.name}: 抽取 {saved_count} 张图片")
    return next_index


def main() -> None:
    """Extract frames from all videos in the configured project directory."""
    if FRAMES_PER_SECOND <= 0:
        raise ValueError("FRAMES_PER_SECOND 必须大于 0")

    source = video_directory()
    target = output_directory()
    if not source.is_dir():
        raise FileNotFoundError(f"视频目录不存在: {source}")
    target.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(source))
    if not videos:
        raise FileNotFoundError(f"目录中没有支持的视频文件: {source}")

    next_index = 1
    for video_path in videos:
        next_index = extract_video(video_path, target, next_index)
    print(f"抽帧完成，输出目录: {target}")


if __name__ == "__main__":
    main()
