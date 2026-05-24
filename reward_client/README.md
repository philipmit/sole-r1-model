# Reward Client

CLI for querying the reward server. Loads frames from one or two video sources (front camera or front+wrist camera), sends them to the reward server over ZMQ (via [agentlace](https://github.com/youliangtan/agentlace)), and prints per-timestep task-progress predictions.


Convenience script. Installs `uv` and dependencies, make a prediction for an "open drawer" example video.

```bash
./run.sh [REWARD SERVER IP]
```

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv). Tested with **Python 3.10** on **Ubuntu 22.04**.

```bash
uv sync
```

## Usage

```bash
# agentview only
python main.py --host localhost --port 8001 \
    --front ../videos/open_drawer/front.webm \
    --task "open the drawer" \
    --video-output out.webm

# agentview + wristview
python main.py --host localhost --port 8001 \
    --front ../videos/open_drawer/front.webm --wrist ../videos/open_drawer/wrist.webm \
    --task "open the drawer" \
    --video-output out.webm

# agentview + wristview, don't use previous task progress (from_zero)
python main.py --host localhost --port 8001 \
    --front ../videos/open_drawer/front.webm --wrist ../videos/open_drawer/wrist.webm \
    --task "open the drawer" --from-zero \
    --video-output out.webm
```

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `localhost` | Reward server IP or hostname |
| `--port` | `8001` | Reward server ZMQ port |
| `--front` | *(required)* | Front-camera source (video, image dir, `.npy`/`.npz`, `.h5`) |
| `--wrist` | — | Wrist-camera source (same formats as `--front`) |
| `--task` | *(required)* | Natural-language task description |
| `--target-fps` | `1.0` | Frames per second to send to the server |
| `--from-zero` | off | Score each frame independently from t=0 |
| `--output-file` | — | Save predictions as JSON |
| `--video-output` | — | Render annotated video |

Run `python main.py --help` for the full list.

## Supported input formats

- Video files: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`
- Image directories (`.jpg`, `.png`, etc.)
- NumPy arrays: `.npy`, `.npz`
- HDF5 files: `.h5`, `.hdf5`
