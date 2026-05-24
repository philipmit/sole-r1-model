"""
Reward client CLI.

Loads frames from one or two video sources, sends them to the reward server via
agentlace (ZMQ req-rep), and prints per-timestep task-progress predictions.

Usage examples
--------------
# agentview only
python main.py --host localhost --port 8001 \
    --front ../example/open_drawer/front.webm \
    --task "open the drawer" \
    --video-output out.webm

# agentview + wristview
python main.py --host localhost --port 8001 \
    --front ../example/open_drawer/front.webm --wrist ../example/open_drawer/wrist.webm \
    --task "open the drawer" \
    --video-output out.webm

# agentview + wristview, don't use previous task progress (from_zero)
python main.py --host localhost --port 8001 \
    --front ../example/open_drawer/front.webm --wrist ../example/open_drawer/wrist.webm \
    --task "open the drawer" --from-zero \
    --video-output out.webm
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

import cv2

from utils import InferenceClient, create_video_with_plot

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
HDF5_EXTENSIONS = {".h5", ".hdf5"}

FRONT_ALIASES = {"agentview", "front", "rgb", "external"}
WRIST_ALIASES = {"wristview", "wrist", "hand"}

DEFAULT_SERVICE_NAME = "qwen_rewards"
DEFAULT_PORT = 8001


def _numeric_sort_key(p: Path) -> tuple:
    """Sort key: extract all digit groups from the stem, then fall back to name."""
    nums = re.findall(r"\d+", p.stem)
    return tuple(int(n) for n in nums) if nums else (p.name,)


def load_frames_from_video(path: Path, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end >= 0 and idx >= end:
            break
        if idx >= start and (idx - start) % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from video: {path}")
    return frames


def load_frames_from_directory(path: Path, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load images from a directory, sorted numerically by filename."""
    from PIL import Image

    candidates = sorted(
        [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=_numeric_sort_key,
    )
    if not candidates:
        raise ValueError(f"No images found in directory: {path}")

    frames = []
    for idx, img_path in enumerate(candidates):
        if end >= 0 and idx >= end:
            break
        if idx >= start and (idx - start) % stride == 0:
            img = Image.open(img_path).convert("RGB")
            frames.append(np.array(img))

    if not frames:
        raise ValueError(f"No frames selected from directory: {path}")
    return frames


def load_frames_from_numpy(path: Path, npz_key: str, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load frames from a .npy or .npz file. Expected shape: (T, H, W, C)."""
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        data = np.load(str(path))
        if npz_key not in data:
            available = list(data.keys())
            raise ValueError(
                f"Key '{npz_key}' not found in {path}. Available: {available}"
            )
        arr = data[npz_key]

    if arr.ndim != 4:
        raise ValueError(f"Expected shape (T,H,W,C), got {arr.shape} from {path}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    indices = range(start, (len(arr) if end < 0 else min(end, len(arr))), stride)
    frames = [arr[i] for i in indices]
    if not frames:
        raise ValueError(f"No frames selected from numpy file: {path}")
    return frames


def load_frames_from_hdf5(path: Path, hdf5_key: str, stride: int, start: int, end: int) -> list[np.ndarray]:
    """Load frames from an HDF5 file. Expected dataset shape: (T, H, W, C)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        if hdf5_key not in f:
            available = list(f.keys())
            raise ValueError(
                f"Key '{hdf5_key}' not found in {path}. Available: {available}"
            )
        arr = f[hdf5_key][:]

    if arr.ndim != 4:
        raise ValueError(f"Expected shape (T,H,W,C), got {arr.shape} from {path}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    indices = range(start, (len(arr) if end < 0 else min(end, len(arr))), stride)
    frames = [arr[i] for i in indices]
    if not frames:
        raise ValueError(f"No frames selected from HDF5 file: {path}")
    return frames


def estimate_source_fps(path: str) -> float | None:
    """
    Read the native frame rate from a video file via OpenCV metadata.
    Returns None for image directories, .npy/.npz, and .h5/.hdf5 inputs.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return float(fps)
    return None


def compute_stride(source_fps: float, target_fps: float) -> int:
    """Return the frame stride needed to sample a source at approximately target_fps."""
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    return max(1, round(source_fps / target_fps))


def load_frames(
    path: str,
    stride: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    npz_key: str = "frames",
    hdf5_key: str = "frames",
) -> list[np.ndarray]:
    """
    Unified frame loader. Returns a list of RGB uint8 HWC numpy arrays.

    Supported sources:
        - Video files:     .mp4 .avi .mov .mkv .webm .m4v
        - Image directory: directory containing .jpg/.png/.bmp/.tiff/.webp files
        - NumPy archive:   .npy  (T,H,W,C)  or  .npz  (key=npz_key)
        - HDF5 file:       .h5 / .hdf5  (dataset key=hdf5_key)
    """
    p = Path(path)

    if p.is_dir():
        return load_frames_from_directory(p, stride, start_frame, end_frame)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    ext = p.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return load_frames_from_video(p, stride, start_frame, end_frame)
    if ext in NUMPY_EXTENSIONS:
        return load_frames_from_numpy(p, npz_key, stride, start_frame, end_frame)
    if ext in HDF5_EXTENSIONS:
        return load_frames_from_hdf5(p, hdf5_key, stride, start_frame, end_frame)

    raise ValueError(
        f"Unsupported file type '{ext}' for path: {path}\n"
        f"Supported: video {sorted(VIDEO_EXTENSIONS)}, "
        f"numpy {sorted(NUMPY_EXTENSIONS)}, hdf5 {sorted(HDF5_EXTENSIONS)}, "
        f"or a directory of images."
    )


def _find_subdir(parent: Path, aliases: set[str]) -> Path | None:
    """Return the first subdirectory of parent whose lower-case name is in aliases."""
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() in aliases:
            return child
    return None


def resolve_views(front_arg: str, wrist_arg: str | None) -> tuple[str, str | None]:
    """
    Resolve front and (optionally) wrist paths.

    If front_arg is a directory containing canonical multi-view subdirectories
    (e.g. agentview/ + wristview/), both are auto-detected and wrist_arg is
    ignored (with a warning if it was also provided).

    Returns (front_path, wrist_path_or_None).
    """
    front_path = Path(front_arg)
    if front_path.is_dir():
        front_sub = _find_subdir(front_path, FRONT_ALIASES)
        wrist_sub = _find_subdir(front_path, WRIST_ALIASES)

        if front_sub is not None and wrist_sub is not None:
            if wrist_arg is not None:
                logging.warning(
                    "Auto-detected multi-view subdirectories inside '%s'; "
                    "ignoring --wrist %s",
                    front_arg,
                    wrist_arg,
                )
            logging.info(
                "Auto-detected views: front='%s', wrist='%s'", front_sub, wrist_sub
            )
            return str(front_sub), str(wrist_sub)

    return front_arg, wrist_arg


def build_payload(
    task: str,
    front_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray] | None,
    from_zero: bool,
    temperature: float,
) -> dict:
    """
    Build the request payload for the reward server.

    The server expects list[list[image]] (episodes × timesteps).
    We wrap the single episode in an outer list.
    """
    external_only = wrist_frames is None
    payload = {
        "task": task,
        "front_images": [front_frames],
        "wrist_images": [wrist_frames if wrist_frames is not None else front_frames],
        "temperature": temperature,
        "from_zero": from_zero,
        "external_only": external_only,
    }
    return payload


def print_results(valid_answers: list[int]) -> None:
    """Print a per-timestep progress table to stdout."""
    col_w = max(len(str(len(valid_answers))), 8)
    header = f"{'Timestep':>{col_w}}   {'Progress':>8}"
    print(header)
    print("-" * len(header))
    for t, v in enumerate(valid_answers):
        print(f"{t:>{col_w}}   {v:>7} %")
    print()
    final = valid_answers[-1]
    print(f"Final reward (last timestep): {final} %")


def save_visualization(
    video_output: str,
    front_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray] | None,
    valid_answers: list[int],
    text_outputs: list[str],
    fps: float,
) -> None:
    """
    Save a side-by-side visualization video:
      [agentview (+ optional wristview stacked vertically)] | [text output] | [reward plot]

    Frame alignment:
      valid_answers[0] == 0 by convention (first frame baseline).
      valid_answers[1..T] are predictions for input frames 0..T-1.
      text_outputs[0..T-1] are the model completions for frames 0..T-1.
      We skip index 0 of valid_answers, aligning predictions 1..T with frames 0..T-1.
    """
    T = len(front_frames)
    # Align predictions: drop the leading 0 baseline entry if present
    answers = list(valid_answers)
    if len(answers) == T + 1:
        answers = answers[1:]
    # Pad or truncate to match frame count
    n = min(T, len(answers))
    frame_list = []
    for i in range(n):
        f = front_frames[i]
        if wrist_frames is not None:
            w = wrist_frames[i]
            # Resize wrist to same height as front before stacking
            if f.shape[0] != w.shape[0]:
                scale = f.shape[0] / w.shape[0]
                w = cv2.resize(w, (int(w.shape[1] * scale), f.shape[0]), interpolation=cv2.INTER_AREA)
            f = np.hstack([f, w])
        frame_list.append(f)

    # Pad text_outputs if shorter
    texts = list(text_outputs) if text_outputs is not None else []
    texts = texts[:n]
    while len(texts) < n:
        texts.append("")

    data_points = answers[:n]

    logging.info("Saving visualization video to: %s", video_output)
    create_video_with_plot(
        output_video_path=video_output,
        frame_list=frame_list,
        frame_desription_list=texts,
        data_points=data_points,
        data_points_env=None,
        fps_=fps,
    )
    logging.info("Visualization video saved.")


def save_results(
    output_file: str,
    task: str,
    front_path: str,
    wrist_path: str | None,
    from_zero: bool,
    external_only: bool,
    data: dict,
) -> None:
    """Save inference results to a JSON file."""
    result = {
        "task": task,
        "front_path": front_path,
        "wrist_path": wrist_path,
        "from_zero": from_zero,
        "external_only": external_only,
        "valid_answers": data["valid_answers"][0].tolist(),
        "text_outputs": data["text_outputs"][0],
        "text_inputs": data["text_inputs"][0],
    }
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    logging.info("Results saved to %s", output_file)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the reward client."""
    parser = argparse.ArgumentParser(
        description="Reward client: send a video to the reward server and get task-progress predictions.",
    )
    # Connection
    parser.add_argument("--host", default="localhost", help="Reward server IP or hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Reward server ZMQ port (default: {DEFAULT_PORT})")
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME, help=f"Server interface name (default: {DEFAULT_SERVICE_NAME})")
    parser.add_argument("--wait-for-server", action="store_true", help="Block until the server is reachable before sending")

    # Video input
    parser.add_argument("--front", required=True, metavar="PATH",
                        help="Front/external camera source: video file, image directory, .npy/.npz, or .h5/.hdf5")
    parser.add_argument("--wrist", default=None, metavar="PATH",
                        help="Wrist camera source (same format as --front). "
                             "If omitted, server runs in external-only mode. "
                             "Auto-detected if --front is a directory containing agentview/ and wristview/ subdirs.")

    # Frequency / frame selection
    parser.add_argument("--target-fps", type=float, default=1.0, metavar="HZ",
                        help="Target sampling frequency in Hz (default: 1.0). "
                             "Determines how many frames per second are sent to the server.")
    parser.add_argument("--source-fps", type=float, default=None, metavar="HZ",
                        help="Source video frequency in Hz. Overrides auto-detection from video metadata. "
                             "Required for image directories, .npy/.npz, and .h5/.hdf5 inputs "
                             "when --target-fps differs from the true capture rate.")
    parser.add_argument("--stride", type=int, default=None, metavar="N",
                        help="Use every N-th frame directly, bypassing frequency-based stride "
                             "computation. Takes precedence over --target-fps / --source-fps.")
    parser.add_argument("--start-frame", type=int, default=0, metavar="N", help="First frame index to use (default: 0)")
    parser.add_argument("--end-frame", type=int, default=-1, metavar="N", help="Exclusive end frame index; -1 means all (default: -1)")
    parser.add_argument("--npz-key", default="frames", help="Array key inside .npz files (default: frames)")
    parser.add_argument("--hdf5-key", default="frames", help="Dataset key inside HDF5 files (default: frames)")

    # Task & inference
    parser.add_argument("--task", required=True, help="Natural-language task description")
    parser.add_argument("--from-zero", action="store_true",
                        help="Predict each timestep independently from t=0 (no carry-forward of previous progress)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")

    # Output
    parser.add_argument("--output-file", default=None, metavar="PATH",
                        help="Optional path to save results as JSON")
    parser.add_argument("--video-output", default=None, metavar="PATH",
                        help="Optional path to save a visualization video (.mp4 or .webm) "
                             "showing the frames, text outputs, and reward plot")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second for the output visualization video (default: 2)")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the reward client: load frames, query the reward server, and report results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    front_path, wrist_path = resolve_views(args.front, args.wrist)

    if args.stride is not None:
        sampling_stride = args.stride
        logging.info("Using explicit stride=%d (--target-fps / --source-fps ignored).", sampling_stride)
    else:
        source_fps = args.source_fps
        if source_fps is None:
            source_fps = estimate_source_fps(front_path)
            if source_fps is not None:
                logging.info("Estimated source FPS from video metadata: %.3g Hz", source_fps)
            else:
                logging.warning(
                    "Cannot estimate source FPS for non-video input '%s'. "
                    "Assuming source FPS == target FPS (stride=1). "
                    "Use --source-fps to set it explicitly.",
                    front_path,
                )
                source_fps = args.target_fps  # stride will be 1
        else:
            logging.info("Using user-provided source FPS: %.3g Hz", source_fps)
        sampling_stride = compute_stride(source_fps, args.target_fps)
        logging.info(
            "Sampling: source=%.3g Hz, target=%.3g Hz → stride=%d "
            "(effective rate ≈ %.3g Hz)",
            source_fps,
            args.target_fps,
            sampling_stride,
            source_fps / sampling_stride,
        )

    load_kwargs = dict(
        stride=sampling_stride,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        npz_key=args.npz_key,
        hdf5_key=args.hdf5_key,
    )

    logging.info("Loading front frames from: %s", front_path)
    front_frames = load_frames(front_path, **load_kwargs)
    logging.info("Loaded %d front frames.", len(front_frames))

    wrist_frames = None
    if wrist_path is not None:
        logging.info("Loading wrist frames from: %s", wrist_path)
        wrist_frames = load_frames(wrist_path, **load_kwargs)
        logging.info("Loaded %d wrist frames.", len(wrist_frames))
        if len(front_frames) != len(wrist_frames):
            logging.error(
                "Frame count mismatch: front has %d frames, wrist has %d frames.",
                len(front_frames),
                len(wrist_frames),
            )
            sys.exit(1)

    logging.info("Connecting to reward server at %s:%d ...", args.host, args.port)
    client = InferenceClient(
        server_ip=args.host,
        port_num=args.port,
        wait_for_server=args.wait_for_server,
    )

    payload = build_payload(
        task=args.task,
        front_frames=front_frames,
        wrist_frames=wrist_frames,
        from_zero=args.from_zero,
        temperature=args.temperature,
    )

    logging.info(
        "Sending %d frame(s), task='%s', from_zero=%s, external_only=%s ...",
        len(front_frames),
        args.task,
        payload["from_zero"],
        payload["external_only"],
    )
    response = client.call(args.service_name, payload)

    if response is None:
        logging.error("No response received from server (timeout or connection error).")
        sys.exit(1)

    if not response.get("success", False):
        logging.error("Server returned an error: %s", response.get("message", "<no message>"))
        sys.exit(1)

    data = response["data"]
    valid_answers = data["valid_answers"][0]

    print()
    print_results(list(valid_answers))

    if args.video_output:
        save_visualization(
            video_output=args.video_output,
            front_frames=front_frames,
            wrist_frames=wrist_frames,
            valid_answers=list(valid_answers),
            text_outputs=data.get("text_outputs", [[]])[0],
            fps=args.fps,
        )

    if args.output_file:
        save_results(
            output_file=args.output_file,
            task=args.task,
            front_path=front_path,
            wrist_path=wrist_path,
            from_zero=args.from_zero,
            external_only=payload["external_only"],
            data=data,
        )


if __name__ == "__main__":
    main()
