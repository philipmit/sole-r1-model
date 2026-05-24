# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import base64
import io
import logging
import re
import threading
from typing import Callable

import cv2
import numpy as np
import torch
from agentlace.zmq_wrapper.req_rep import ReqRepServer
from PIL import Image

from reward_server.constants import QUESTION_TEMPLATE, SYSTEM_PROMPT_TEMPLATE


class InferenceServer:
    def __init__(self, port_num: int, log_level=logging.INFO):
        """
        Create a server with the given port number and interface names
        """

        def __parser_cb(payload: dict) -> dict:
            # if is list_interfaces type
            if payload.get("type") == "list_interfaces":
                return {"interfaces": list(self.interfaces.keys())}
            elif payload.get("type") == "call_interface":
                interface_name = payload.get("interface")
                if interface_name and interface_name in self.interfaces:
                    return self.interfaces[interface_name](payload["payload"])
            return {"success": False, "message": "Invalid interface or payload"}

        self.server = ReqRepServer(port_num, __parser_cb, log_level=log_level)
        self.interfaces = {}

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
            :param threaded: Whether to start the server in a separate thread
        """
        logging.info("Starting Inference Server.")
        if threaded:
            self.thread = threading.Thread(target=self.server.run)
            self.thread.start()
        else:
            self.server.run()

    def register_interface(self, name: str, callback: Callable):
        """
        Registers the callback function for the interface
            :param name: Name of the interface
            :param callback: Callback function for the interface
        """
        self.interfaces[name] = callback

    def stop(self):
        """Stop the server"""
        self.server.stop()


def resize_with_padding(img: np.ndarray, size: int = 384) -> np.ndarray:
    """Aspect ratio preserving image resize with padding.

    Args:
        img: Input image.
        size: Desired output size (size x size).

    Returns:
        Resized image with padding.
    """
    h, w = img.shape[:2]
    # Determine scaling factor so max dimension == size
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Resize with preserved aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create a black canvas 384x384
    output = np.zeros((size, size, 3), dtype=np.uint8)
    # Center the resized image on the canvas
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    output[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return output


def create_composite_frame(
    first_frame_wrist_view: np.ndarray,
    first_frame_external_view: np.ndarray,
    frame0_wrist_view: np.ndarray,
    frame0_external_view: np.ndarray,
    frame1_wrist_view: np.ndarray,
    frame1_external_view: np.ndarray,
    from_zero: bool = False,
    external_only: bool = False,
    size: int = 384,
    padding: int = 5,
) -> np.ndarray:
    """Create composite image frame for VLM input."""
    first_imgs = [first_frame_wrist_view, first_frame_external_view]
    imgs0 = [frame0_wrist_view, frame0_external_view]
    imgs2 = [frame1_wrist_view, frame1_external_view]

    for i in range(len(first_imgs)):
        first_imgs[i] = resize_with_padding(first_imgs[i], size)
        imgs0[i] = resize_with_padding(imgs0[i], size)
        imgs2[i] = resize_with_padding(imgs2[i], size)

    col_pad = np.zeros((size, padding, 3), dtype=np.uint8)
    if not from_zero:
        bottom_row = np.hstack([first_imgs[0], col_pad, imgs0[0], col_pad, imgs2[0]])
        top_row = np.hstack(
            [
                first_imgs[1],
                col_pad,
                imgs0[1],
                col_pad,
                imgs2[1],
            ]
        )
    else:
        bottom_row = np.hstack([first_imgs[0], col_pad, imgs2[0]])
        top_row = np.hstack([first_imgs[1], col_pad, imgs2[1]])
    # Create horizontal (row) padding between timesteps
    # row_pad = np.zeros((padding, size, 3), dtype=np.uint8)
    full_width = top_row.shape[1]
    row_pad = np.zeros((padding, full_width, 3), dtype=np.uint8)

    if external_only:
        return top_row
    # Now stack horizontally instead of vertically
    return np.vstack([top_row, row_pad, bottom_row])


def image_to_base64(img: np.ndarray, quality: int = 90) -> str:
    """Convert a numpy array image to base64 encoded string."""
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_images(images: list[list[np.ndarray]], quality: int = 90) -> list[list[str]]:
    """Encode a list of images to base64 strings."""
    encoded_images = []
    for ep_images in images:
        encoded_ep = []
        for img in ep_images:
            encoded_img = image_to_base64(img, quality=quality)
            encoded_ep.append(encoded_img)
        encoded_images.append(encoded_ep)
    return encoded_images


def decode_compressed_image(png_bytes: np.uint8) -> np.ndarray:
    """Decode PNG bytes into a numpy array image."""
    stream = io.BytesIO(png_bytes.tobytes())
    img = Image.open(stream)
    # Ensure that image data is fully loaded.
    img.load()
    return np.array(img)


def count_images(images: list[list]) -> tuple[int, list[int]]:
    """Count images and validate consistent episode counts and lengths."""
    num_episodes = len(images)
    episode_lengths = [len(epi) for epi in images]
    return num_episodes, episode_lengths


def count_and_validate_images(
    front_images: list[list], wrist_images: list[list]
) -> tuple[int, list[int]]:
    """Count images and validate consistent episode counts and lengths."""
    assert len(front_images) == len(
        wrist_images
    ), f"Length mismatch: {len(front_images)} vs {len(wrist_images)}"
    num_episodes = len(front_images)
    episode_lengths = []
    for epi_idx in range(num_episodes):
        front_imgs = front_images[epi_idx]
        wrist_imgs = wrist_images[epi_idx]
        assert len(front_imgs) == len(
            wrist_imgs
        ), f"Length mismatch in episode {epi_idx}: {len(front_imgs)} vs {len(wrist_imgs)}"
        episode_lengths.append(len(front_imgs))
    return num_episodes, episode_lengths


def process_images(images: list[list]) -> list[list[np.ndarray]]:
    """Decode and validate images from bytes to numpy arrays.

    Args:
        images: List of episodes, each containing a list of image bytes.

    Returns:
        Processed images as numpy array in HWC uint8 format.
    """
    processed_images = []
    for ep_images in images:
        processed_ep = []
        for img_bytes in ep_images:
            img = decode_and_validate_image(img_bytes)
            processed_ep.append(img)
        processed_images.append(processed_ep)
    return processed_images


def decode_and_validate_image(image: np.ndarray | torch.Tensor | bytes) -> np.ndarray:
    """Decode and validate image input into a numpy array in HWC uint8 format.

    Case 1: image is bytes (compressed PNG/JPEG).
    Case 2: image is torch.Tensor (uint8 or float32).
    Case 3: image is uint8 np.ndarray of bytes (compressed PNG/JPEG).
    Case 4: image is np.ndarray in CHW or HWC format, uint8 or float32.
    """
    if isinstance(image, bytes):
        # Decode image bytes.
        image = decode_compressed_image(np.frombuffer(image, dtype=np.uint8))
    if isinstance(image, torch.Tensor):
        # Torch into numpy.
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 1:
            # Decode image bytes.
            image = decode_compressed_image(image)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            # CHW to HWC.
            image = np.moveaxis(image, 0, -1)
        if image.dtype != np.uint8:
            # E.g., float32 to uint8.
            if image.max() <= 1.0:
                # [0, 1] to [0, 255].
                # Note that this could fail with black images.
                # So better send uint8 images.
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)

    assert image.ndim == 3, f"Image must be HWC format, got shape {image.shape}."
    assert image.shape[2] in [
        1,
        3,
    ], f"Image must have 1 or 3 channels, got shape {image.shape}."
    assert (
        image.dtype == np.uint8
    ), f"Image must be uint8 type, got dtype {image.dtype}. This is an internal error."

    return image


def prepare_vlm_images(
    front_images: list[list[np.ndarray]],
    wrist_images: list[list[np.ndarray]],
    num_episodes: int,
    episode_lengths: list[int],
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    factor: int = 28,
    from_zero: bool = False,
    external_only: bool = False,
) -> list[list[Image.Image]]:
    """Prepare composite VLM images for each episode and timestep."""
    from qwen_vl_utils import smart_resize

    vlm_images = []
    for ep_idx in range(num_episodes):
        vlm_images_epi = []
        for t in range(1, episode_lengths[ep_idx]):
            composite_frame = create_composite_frame(
                first_frame_wrist_view=wrist_images[ep_idx][0],
                first_frame_external_view=front_images[ep_idx][0],
                frame0_wrist_view=wrist_images[ep_idx][t - 1],
                frame0_external_view=front_images[ep_idx][t - 1],
                frame1_wrist_view=wrist_images[ep_idx][t],
                frame1_external_view=front_images[ep_idx][t],
                from_zero=from_zero,
                external_only=external_only,
            )
            composite_frame = Image.fromarray(composite_frame)

            width, height = composite_frame.size
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            composite_frame = composite_frame.resize((resized_width, resized_height))

            vlm_images_epi.append(composite_frame)
        vlm_images.append(vlm_images_epi)
    return vlm_images


def make_conversation_image(
    question: str,
    system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE,
    question_template: str = QUESTION_TEMPLATE,
) -> list[dict]:
    """Create conversation format input for VLMs with image and text."""
    return [
        {
            "role": "system",
            "content": system_prompt_template.format(question=question),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": question_template.format(question=question),
                },
            ],
        },
    ]


def assemble_output_batch(outputs: list, indices: list[int], video_count: int) -> list:
    """Assemble outputs into a batch aligned with input video indices.

    Some values can be None if the episode is shorter than others.
    """
    output_batch = [None] * video_count
    for out, idx in zip(outputs, indices):
        output_batch[idx] = out
    return output_batch


def get_output_across_videos(
    video_count: int,
    text_input_list_batch: list[list],
    text_output_list_batch: list[list],
    example_list_batch: list[list],
    answer_list_batch: list[list],
) -> tuple[list[list], list[list], list[list], list[list]]:
    """Organize outputs on a per-episode basis.

    This function handles cases where some episodes are shorter than others.
    This manifests as None values in the input lists for timesteps beyond the episode length.
    We double-check that once we see a None the episode is done and all subsequent values are also None.
    We also double-check that the four input lists are aligned in terms of None values.
    """
    text_output_list_list = []
    text_input_list_list = []
    example_list_list = []
    answer_list_list = []
    for video_idx in range(video_count):
        text_inputs, text_outputs, examples, answers = [], [], [], []
        seen_none = False
        for ti, to, ex, an in zip(
            [x[video_idx] for x in text_input_list_batch],
            [x[video_idx] for x in text_output_list_batch],
            [x[video_idx] for x in example_list_batch],
            [x[video_idx] for x in answer_list_batch],
        ):
            none_mask = (ti is None, to is None, ex is None, an is None)
            assert all(none_mask) or not any(
                none_mask
            ), f"Misaligned Nones at video {video_idx}: {none_mask}"

            is_none = ti is None
            if seen_none and not is_none:
                raise AssertionError(
                    f"Video {video_idx}: Non-None value appeared after None. "
                    f"Once None appears, all subsequent values must be None."
                )

            if is_none:
                seen_none = True
            else:
                # Only append if not None
                text_inputs.append(ti)
                text_outputs.append(to)
                examples.append(ex)
                answers.append(an)
        text_output_list_list.append(text_outputs)
        text_input_list_list.append(text_inputs)
        example_list_list.append(examples)
        answer_list_list.append(answers)
    return (
        text_output_list_list,
        text_input_list_list,
        example_list_list,
        answer_list_list,
    )


def get_answer_from_completion(completion):
    """Extract task progress answer from model completion string."""
    answer = ""
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, completion, re.DOTALL)
    if not answer_match:
        answer_pattern = r"<answer>(.*?)</answer"
        answer_match = re.search(answer_pattern, completion, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        answer = answer.replace("%", "")
        try:
            answer_int = int(answer)
            if answer_int < -100:
                answer = "-100"
            elif answer_int > 100:
                answer = "100"
        except Exception as e:
            answer_int = 0

    return answer
