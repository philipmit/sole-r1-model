# Reward Server

Predict dense reward functions for robotic videos. This package either starts a self-hosted VLM or queries GPT and Gemini.

Tested with **Python 3.10** on **Ubuntu 22.04** (CUDA 12.1 for the SOLE-R1 server).

## Run SOLE-R1 server

The server requires a single H100 (or comparable) GPU.

1. Download the model checkpoint from HuggingFace (or point to a local fine-tuned checkpoint).

2. Run the launch script, passing the path to the checkpoint:

```bash
bash run.sh --checkpoint-path /path/to/checkpoint
```

The script installs all dependencies via [uv](https://github.com/astral-sh/uv) and starts the
reward server on port 8001 (override with `--port <port>`).

Optional: to skip the system-dependency installation step (e.g., in a pre-configured environment),
you can install the Python package manually and run the server directly:

```bash
uv sync
uv run python src/reward_server/main.py checkpoint_path=/path/to/checkpoint
```

## Run GPT or Gemini

Execute `python src/reward_server/gpt.py` for a GPT-5 server and `python src/reward_server/gemini.py` for a Gemini-3-Pro server with low thinking.
These servers have a simplified API that only accepts `front_images` and `task`.

## Run with Docker (SOLE-R1)

A `Dockerfile` is provided for running the SOLE-R1 server in a container. Build and run as follows:

```bash
# Build the image
docker build -t sole-r1-server .

# Run the server, mounting your checkpoint directory and exposing the ZMQ port
docker run --gpus all --rm \
    -v /path/to/checkpoint:/checkpoint \
    -p 8001:8001 \
    sole-r1-server \
    uv run python src/reward_server/main.py checkpoint_path=/checkpoint
```

The base image is `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`, so a compatible NVIDIA driver and the NVIDIA Container Toolkit are required on the host.

## Server API

```
Payload:
    task: str, the task name
    front_images: a list of episodes, each episode is a list of front images.
        Images can be formatted as numpy or torch arrays (uint8 or float32) or
        as a list or an array of compressed bytes (e.g., PNG or JPEG).
    wrist_images: a list of episodes, each episode is a list of wrist images.
        Save formatting as above.
    temperature (optional): float, temperature for sampling. Defaults to 1.0.
    from_zero (optional): bool, whether to predict progress from zero at each time step.
        Defaults to False.

Returns:
    success: bool, whether the operation was successful.
    message (optional): str, error message if not successful.
    data (optional): dict, if successful, contains:
        - "valid_answers": list of np.ndarray, each array contains the predicted
            task progress percentages for each timestep in the episode.
        - "text_outputs": list of list of str, the raw text outputs from the model
            for each episode and timestep.
        - "text_inputs": list of list of str, the text inputs to the model
            for each episode and timestep.
```
