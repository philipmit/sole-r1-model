# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Run as `pytest reward_server/src/reward_server/tests/test_utils.py`. Set `VISUALIZE=True` to see image outputs.

VISUALIZE = False
if VISUALIZE:
    import os

    # Suppress Qt Wayland warnings (e.g., requestActivate() not supported)
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland=false")
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
    import matplotlib

    matplotlib.use("QtAgg")

import io

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from reward_server.utils import (
    assemble_output_batch,
    count_and_validate_images,
    create_composite_frame,
    decode_and_validate_image,
    decode_compressed_image,
    get_answer_from_completion,
    get_output_across_videos,
    prepare_vlm_images,
    resize_with_padding,
)


def test_resize_with_padding_square_image():
    """Test resizing a square image - should scale down to size x size with no padding."""
    # Create a 512x512 red image
    img = np.ones((512, 512, 3), dtype=np.uint8) * [255, 0, 0]
    img = img.astype(np.uint8)

    result = resize_with_padding(img, size=384)
    if VISUALIZE:
        plt.imshow(result)
        plt.title("Resized Square Image")
        plt.show()

    # Check output shape
    assert result.shape == (384, 384, 3)

    # For a square image, the entire output should be the resized image (no black padding)
    # Check that center pixels are red (not black)
    center_pixel = result[192, 192]
    assert np.array_equal(center_pixel, [255, 0, 0])


def test_resize_with_padding_wide_image():
    """Test resizing a wide (landscape) image - should have padding on top and bottom."""
    # Create a 200x800 green image (wide)
    img = np.ones((200, 800, 3), dtype=np.uint8) * [0, 255, 0]
    img = img.astype(np.uint8)

    result = resize_with_padding(img, size=384)
    if VISUALIZE:
        plt.imshow(result)
        plt.title("Resized Wide Image")
        plt.show()

    # Check output shape
    assert result.shape == (384, 384, 3)

    # Width should be scaled to 384, height to 96 (200 * 384/800)
    # So we expect padding on top and bottom
    # Top padding should be (384 - 96) // 2 = 144

    # Check top padding is black
    assert np.array_equal(result[0, 192], [0, 0, 0])

    # Check center vertical position has the green image
    center_y = 192  # Middle of 384
    assert np.array_equal(result[center_y, 192], [0, 255, 0])

    # Check bottom padding is black
    assert np.array_equal(result[383, 192], [0, 0, 0])


def test_resize_with_padding_tall_image():
    """Test resizing a tall (portrait) image - should have padding on left and right."""
    # Create a 800x200 blue image (tall)
    img = np.ones((800, 200, 3), dtype=np.uint8) * [0, 0, 255]
    img = img.astype(np.uint8)

    result = resize_with_padding(img, size=384)
    if VISUALIZE:
        plt.imshow(result)
        plt.title("Resized Tall Image")
        plt.show()

    # Check output shape
    assert result.shape == (384, 384, 3)

    # Height should be scaled to 384, width to 96 (200 * 384/800)
    # So we expect padding on left and right
    # Left padding should be (384 - 96) // 2 = 144

    # Check left padding is black
    assert np.array_equal(result[192, 0], [0, 0, 0])

    # Check center horizontal position has the blue image
    center_x = 192  # Middle of 384
    assert np.array_equal(result[192, center_x], [0, 0, 255])

    # Check right padding is black
    assert np.array_equal(result[192, 383], [0, 0, 0])


def test_resize_with_padding_custom_size():
    """Test resizing with a custom size parameter."""
    # Create a 600x400 yellow image
    img = np.ones((600, 400, 3), dtype=np.uint8) * [255, 255, 0]
    img = img.astype(np.uint8)

    result = resize_with_padding(img, size=256)
    if VISUALIZE:
        plt.imshow(result)
        plt.title("Resized Custom Size Image")
        plt.show()

    # Check output shape
    assert result.shape == (256, 256, 3)

    # Max dimension is 600, so scale = 256/600
    # new_h = 256, new_w = int(400 * 256/600) = 170
    # x_offset = (256 - 170) // 2 = 43

    # Check that the center has the yellow image
    assert np.array_equal(result[128, 128], [255, 255, 0])


def test_create_composite_frame_default():
    """Test create_composite_frame with default parameters (from_zero=False)."""
    # Create distinct colored frames for testing
    size = 100
    first_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [255, 0, 0]).astype(
        np.uint8
    )  # Red
    first_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 255, 0]).astype(
        np.uint8
    )  # Green
    frame0_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [0, 0, 255]).astype(
        np.uint8
    )  # Blue
    frame0_external = (np.ones((size, size, 3), dtype=np.uint8) * [255, 255, 0]).astype(
        np.uint8
    )  # Yellow
    frame1_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [255, 0, 255]).astype(
        np.uint8
    )  # Magenta
    frame1_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 255, 255]).astype(
        np.uint8
    )  # Cyan

    composite = create_composite_frame(
        first_wrist,
        first_external,
        frame0_wrist,
        frame0_external,
        frame1_wrist,
        frame1_external,
        from_zero=False,
    )

    if VISUALIZE:
        plt.imshow(composite)
        plt.title("Composite Frame - Default (3 timesteps)")
        plt.show()

    # With from_zero=False, we have 3 timesteps (first, frame0, frame1)
    # Each resized to 384x384, with 5px padding between columns
    # Expected width: 384 * 3 + 5 * 2 = 1162
    # Expected height: 384 * 2 + 5 = 773 (top row + padding + bottom row)
    assert composite.shape == (773, 1162, 3)

    # Check that composite has 3 channels (RGB)
    assert composite.shape[2] == 3


def test_create_composite_frame_from_zero():
    """Test create_composite_frame with from_zero=True (2 timesteps only)."""
    size = 100
    first_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [255, 0, 0]).astype(
        np.uint8
    )  # Red
    first_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 255, 0]).astype(
        np.uint8
    )  # Green
    frame0_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [0, 0, 255]).astype(
        np.uint8
    )  # Blue
    frame0_external = (np.ones((size, size, 3), dtype=np.uint8) * [255, 255, 0]).astype(
        np.uint8
    )  # Yellow
    frame1_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [255, 0, 255]).astype(
        np.uint8
    )  # Magenta
    frame1_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 255, 255]).astype(
        np.uint8
    )  # Cyan

    composite = create_composite_frame(
        first_wrist,
        first_external,
        frame0_wrist,
        frame0_external,
        frame1_wrist,
        frame1_external,
        from_zero=True,
    )

    if VISUALIZE:
        plt.imshow(composite)
        plt.title("Composite Frame - from_zero=True (2 timesteps)")
        plt.show()

    # With from_zero=True, we have 2 timesteps (first, frame1)
    # Expected width: 384 * 2 + 5 * 1 = 773
    # Expected height: 384 * 2 + 5 = 773 (top row + padding + bottom row)
    assert composite.shape == (773, 773, 3)


def test_create_composite_frame_different_aspect_ratios():
    """Test create_composite_frame with images of different aspect ratios."""
    # Create images with different aspect ratios to ensure padding works correctly
    wrist_wide = (np.ones((200, 800, 3), dtype=np.uint8) * [255, 0, 0]).astype(
        np.uint8
    )  # Wide
    external_tall = (np.ones((800, 200, 3), dtype=np.uint8) * [0, 255, 0]).astype(
        np.uint8
    )  # Tall

    composite = create_composite_frame(
        wrist_wide,
        external_tall,
        wrist_wide,
        external_tall,
        wrist_wide,
        external_tall,
        from_zero=False,
    )

    if VISUALIZE:
        plt.imshow(composite)
        plt.title("Composite Frame - Different Aspect Ratios")
        plt.show()

    # Verify expected dimensions
    assert composite.shape == (773, 1162, 3)

    # Verify no NaN or invalid values
    assert not np.isnan(composite).any()
    assert composite.dtype == np.uint8


def test_create_composite_frame_image_placement():
    """Test that images are correctly placed in the composite."""
    size = 100
    # Create frames with distinct pixel patterns to verify placement
    first_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [200, 0, 0]).astype(
        np.uint8
    )  # Red
    first_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 200, 0]).astype(
        np.uint8
    )  # Green
    frame0_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [0, 0, 200]).astype(
        np.uint8
    )  # Blue
    frame0_external = (np.ones((size, size, 3), dtype=np.uint8) * [200, 200, 0]).astype(
        np.uint8
    )  # Yellow
    frame1_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [200, 0, 200]).astype(
        np.uint8
    )  # Magenta
    frame1_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 200, 200]).astype(
        np.uint8
    )  # Cyan

    composite = create_composite_frame(
        first_wrist,
        first_external,
        frame0_wrist,
        frame0_external,
        frame1_wrist,
        frame1_external,
        from_zero=False,
    )

    if VISUALIZE:
        plt.imshow(composite)
        plt.title("Composite Frame - Image Placement Verification")
        plt.show()

    # Top row should be external views (green, yellow, cyan)
    # Bottom row should be wrist views (red, blue, magenta)
    # Check that top-left section contains green (first_external)
    top_left_center = composite[192, 192]  # Center of first 384x384 block
    assert top_left_center[1] > 150  # Strong green component

    # Check that bottom-left section contains red (first_wrist)
    bottom_left_center = composite[384 + 5 + 192, 192]  # Below padding
    assert bottom_left_center[0] > 150  # Strong red component

    # Check that top-middle section contains yellow (frame0_external)
    top_middle_center = composite[192, 384 + 5 + 192]
    assert (
        top_middle_center[0] > 150 and top_middle_center[1] > 150
    )  # Strong red and green

    # Check that bottom-middle section contains blue (frame0_wrist)
    bottom_middle_center = composite[384 + 5 + 192, 384 + 5 + 192]
    assert bottom_middle_center[2] > 150  # Strong blue component

    # Check that top-right section contains cyan (frame1_external)
    top_right_center = composite[192, 2 * (384 + 5) + 192]
    assert (
        top_right_center[1] > 150 and top_right_center[2] > 150
    )  # Strong green and blue

    # Check that bottom-right section contains magenta (frame1_wrist)
    bottom_right_center = composite[384 + 5 + 192, 2 * (384 + 5) + 192]
    assert (
        bottom_right_center[0] > 150 and bottom_right_center[2] > 150
    )  # Strong red and blue


def test_count_and_validate_valid_data():
    """Test count_and_validate with valid matching data."""
    # Create test data with 3 episodes of varying lengths
    front_images = [
        [[1, 2, 3], [4, 5, 6]],  # Episode 0: 2 frames
        [[7, 8, 9], [10, 11, 12], [13, 14, 15]],  # Episode 1: 3 frames
        [[16, 17, 18]],  # Episode 2: 1 frame
    ]
    wrist_images = [
        [[19, 20, 21], [22, 23, 24]],  # Episode 0: 2 frames (matches front)
        [
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
        ],  # Episode 1: 3 frames (matches front)
        [[34, 35, 36]],  # Episode 2: 1 frame (matches front)
    ]

    num_episodes, episode_lengths = count_and_validate_images(
        front_images, wrist_images
    )

    assert num_episodes == 3
    assert episode_lengths == [2, 3, 1]


def test_count_and_validate_length_mismatch():
    """Test count_and_validate raises AssertionError on length mismatch."""
    # Test case 1: Different number of episodes
    front_images_1 = [[[1, 2, 3]], [[4, 5, 6]]]
    wrist_images_1 = [[[7, 8, 9]]]  # Only 1 episode instead of 2

    with pytest.raises(AssertionError, match="Length mismatch: 2 vs 1"):
        count_and_validate_images(front_images_1, wrist_images_1)

    # Test case 2: Different number of frames within an episode
    front_images_2 = [
        [[1, 2, 3], [4, 5, 6]],  # Episode 0: 2 frames
        [[7, 8, 9]],  # Episode 1: 1 frame
    ]
    wrist_images_2 = [
        [[10, 11, 12], [13, 14, 15]],  # Episode 0: 2 frames (matches)
        [[16, 17, 18], [19, 20, 21], [22, 23, 24]],  # Episode 1: 3 frames (mismatch!)
    ]

    with pytest.raises(AssertionError, match="Length mismatch in episode 1: 1 vs 3"):
        count_and_validate_images(front_images_2, wrist_images_2)


def test_decode_compressed_image():
    """Test decoding PNG bytes into a numpy array."""

    # Create a simple 10x10 RGB test image with distinct colors
    original_array = np.zeros((10, 10, 3), dtype=np.uint8)
    original_array[:5, :5] = [255, 0, 0]  # Red in top-left quadrant
    original_array[:5, 5:] = [0, 255, 0]  # Green in top-right quadrant
    original_array[5:, :5] = [0, 0, 255]  # Blue in bottom-left quadrant
    original_array[5:, 5:] = [255, 255, 0]  # Yellow in bottom-right quadrant

    # Convert to PNG bytes
    img = Image.fromarray(original_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    # Decode using the function under test
    decoded_array = decode_compressed_image(png_bytes)

    # Verify the decoded image matches the original
    assert decoded_array.shape == original_array.shape
    assert decoded_array.dtype == np.uint8
    assert np.array_equal(decoded_array, original_array)


def test_decode_and_validate_image_torch():
    """Test decode_and_validate_image with a torch.Tensor input."""
    # Create a 10x10 RGB test image as torch tensor in CHW format
    torch_img = torch.zeros((3, 10, 10), dtype=torch.uint8)
    torch_img[0, :5, :5] = 255  # Red in top-left
    torch_img[1, :5, 5:] = 255  # Green in top-right
    torch_img[2, 5:, :5] = 255  # Blue in bottom-left
    torch_img[0:2, 5:, 5:] = 255  # Yellow in bottom-right

    # Decode and validate
    decoded = decode_and_validate_image(torch_img)

    # Verify output format
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (10, 10, 3)  # HWC format
    assert decoded.dtype == np.uint8

    # Verify pixel values (CHW -> HWC conversion)
    assert np.array_equal(decoded[0, 0], [255, 0, 0])  # Red
    assert np.array_equal(decoded[0, 9], [0, 255, 0])  # Green
    assert np.array_equal(decoded[9, 0], [0, 0, 255])  # Blue
    assert np.array_equal(decoded[9, 9], [255, 255, 0])  # Yellow


def test_decode_and_validate_image_numpy():
    """Test decode_and_validate_image with a numpy array input in HWC format."""
    # Create a 10x10 RGB test image as numpy array in HWC format
    numpy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    numpy_img[:5, :5] = [255, 0, 0]  # Red
    numpy_img[:5, 5:] = [0, 255, 0]  # Green
    numpy_img[5:, :5] = [0, 0, 255]  # Blue
    numpy_img[5:, 5:] = [255, 255, 0]  # Yellow

    # Decode and validate
    decoded = decode_and_validate_image(numpy_img)

    # Verify output format
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (10, 10, 3)
    assert decoded.dtype == np.uint8

    # Verify pixel values remain the same
    assert np.array_equal(decoded, numpy_img)


def test_decode_and_validate_image_numpy_bytes():
    """Test decode_and_validate_image with a numpy array of PNG bytes."""
    # Create a simple 10x10 RGB test image
    original_array = np.zeros((10, 10, 3), dtype=np.uint8)
    original_array[:5, :5] = [255, 0, 0]  # Red
    original_array[:5, 5:] = [0, 255, 0]  # Green
    original_array[5:, :5] = [0, 0, 255]  # Blue
    original_array[5:, 5:] = [255, 255, 0]  # Yellow

    # Convert to PNG bytes as numpy array
    img = Image.fromarray(original_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_bytes_numpy = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    # Decode and validate
    decoded = decode_and_validate_image(png_bytes_numpy)

    # Verify output format
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (10, 10, 3)
    assert decoded.dtype == np.uint8

    # Verify pixel values
    assert np.array_equal(decoded, original_array)


def test_decode_and_validate_image_bytes():
    """Test decode_and_validate_image with Python bytes input."""
    # Create a simple 10x10 RGB test image
    original_array = np.zeros((10, 10, 3), dtype=np.uint8)
    original_array[:5, :5] = [255, 0, 0]  # Red
    original_array[:5, 5:] = [0, 255, 0]  # Green
    original_array[5:, :5] = [0, 0, 255]  # Blue
    original_array[5:, 5:] = [255, 255, 0]  # Yellow

    # Convert to PNG bytes
    img = Image.fromarray(original_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()  # Python bytes object

    # Decode and validate
    decoded = decode_and_validate_image(png_bytes)

    # Verify output format
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (10, 10, 3)
    assert decoded.dtype == np.uint8

    # Verify pixel values
    assert np.array_equal(decoded, original_array)


def test_prepare_vlm_images():
    """Test prepare_vlm_images creates proper VLM composite images for multiple episodes."""
    # Create test data with 2 episodes of different lengths
    # Episode 0: 3 frames (will generate 2 VLM images from t=1 and t=2)
    # Episode 1: 4 frames (will generate 3 VLM images from t=1, t=2, t=3)

    # Create distinct colored frames for easy visual verification
    def create_test_image(color_rgb, size=(100, 100)):
        """Helper to create a solid color test image."""
        return (np.ones((size[0], size[1], 3), dtype=np.uint8) * color_rgb).astype(
            np.uint8
        )

    # Episode 0: 3 frames with different colors
    episode_0_front = [
        create_test_image([255, 0, 0]),  # Red - t=0
        create_test_image([0, 255, 0]),  # Green - t=1
        create_test_image([0, 0, 255]),  # Blue - t=2
    ]
    episode_0_wrist = [
        create_test_image([255, 255, 0]),  # Yellow - t=0
        create_test_image([255, 0, 255]),  # Magenta - t=1
        create_test_image([0, 255, 255]),  # Cyan - t=2
    ]

    # Episode 1: 4 frames with different colors
    episode_1_front = [
        create_test_image([128, 0, 0]),  # Dark red - t=0
        create_test_image([0, 128, 0]),  # Dark green - t=1
        create_test_image([0, 0, 128]),  # Dark blue - t=2
        create_test_image([128, 128, 0]),  # Dark yellow - t=3
    ]
    episode_1_wrist = [
        create_test_image([128, 0, 128]),  # Dark magenta - t=0
        create_test_image([0, 128, 128]),  # Dark cyan - t=1
        create_test_image([128, 128, 128]),  # Gray - t=2
        create_test_image([64, 64, 64]),  # Dark gray - t=3
    ]

    front_images = [episode_0_front, episode_1_front]
    wrist_images = [episode_0_wrist, episode_1_wrist]
    num_episodes = 2
    episode_lengths = [3, 4]

    # Call prepare_vlm_images
    vlm_images = prepare_vlm_images(
        front_images=front_images,
        wrist_images=wrist_images,
        num_episodes=num_episodes,
        episode_lengths=episode_lengths,
    )

    # Verify output structure
    assert len(vlm_images) == num_episodes, "Should have 2 episodes"
    assert len(vlm_images[0]) == 2, "Episode 0 should have 2 VLM images (from t=1,2)"
    assert len(vlm_images[1]) == 3, "Episode 1 should have 3 VLM images (from t=1,2,3)"

    # Verify all outputs are PIL Images
    for ep_idx, vlm_epi in enumerate(vlm_images):
        for t_idx, img in enumerate(vlm_epi):
            assert isinstance(
                img, Image.Image
            ), f"Episode {ep_idx}, timestep {t_idx} should be PIL Image"
            # Verify images are not empty
            assert img.size[0] > 0 and img.size[1] > 0, "Image should have valid size"

    # Verify image dimensions are reasonable (after smart_resize)
    # The composite frames should be resized by smart_resize
    for ep_idx, vlm_epi in enumerate(vlm_images):
        for t_idx, img in enumerate(vlm_epi):
            width, height = img.size
            # smart_resize should produce dimensions that are multiples of factor (28)
            assert (
                width % 28 == 0
            ), f"Width {width} should be multiple of factor=28 after smart_resize"
            assert (
                height % 28 == 0
            ), f"Height {height} should be multiple of factor=28 after smart_resize"

    # Optional visualization
    if VISUALIZE:
        # Visualize VLM images from both episodes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("prepare_vlm_images Test Output", fontsize=16)

        # Display episode 0 images (2 timesteps)
        for t_idx in range(2):
            axes[0, t_idx].imshow(vlm_images[0][t_idx])
            axes[0, t_idx].set_title(f"Episode 0, VLM Image t={t_idx+1}")
            axes[0, t_idx].axis("off")
        # Hide the third subplot in first row
        axes[0, 2].axis("off")

        # Display episode 1 images (3 timesteps)
        for t_idx in range(3):
            axes[1, t_idx].imshow(vlm_images[1][t_idx])
            axes[1, t_idx].set_title(f"Episode 1, VLM Image t={t_idx+1}")
            axes[1, t_idx].axis("off")

        plt.tight_layout()
        plt.show()

    # Additional verification: Check that image content is not all zeros
    for ep_idx, vlm_epi in enumerate(vlm_images):
        for t_idx, img in enumerate(vlm_epi):
            img_array = np.array(img)
            assert (
                img_array.max() > 0
            ), f"Episode {ep_idx}, timestep {t_idx} should not be all black"


def test_assemble_output_batch_all_positions():
    """Test assemble_output_batch when all positions are filled."""
    # Create test outputs and indices that fill all positions
    outputs = ["output_0", "output_1", "output_2"]
    indices = [0, 1, 2]
    video_count = 3

    # Call the function
    result = assemble_output_batch(outputs, indices, video_count)

    # Verify the result
    assert len(result) == video_count
    assert result[0] == "output_0"
    assert result[1] == "output_1"
    assert result[2] == "output_2"
    # Verify no None values
    assert all(item is not None for item in result)


def test_assemble_output_batch_missing():
    """Test assemble_output_batch with basic functionality."""
    # Create test outputs and indices
    outputs = ["output_0", "output_1", "output_2"]
    indices = [0, 2, 4]
    video_count = 5

    # Call the function
    result = assemble_output_batch(outputs, indices, video_count)

    # Verify the result
    assert len(result) == video_count
    assert result[0] == "output_0"
    assert result[1] is None  # Not assigned
    assert result[2] == "output_1"
    assert result[3] is None  # Not assigned
    assert result[4] == "output_2"


def test_get_output_across_videos_aligned_nones():
    """Test get_output_across_videos with properly aligned None values.

    Once None appears for a video, all subsequent values should be None,
    and we stop appending (None values are not included in the output).
    """
    # Create test data: 2 videos, 3 batches
    # Video 0: has values in batch 0, then None in batches 1 and 2 (shorter video)
    # Video 1: has values in all batches (longer video)
    text_input_list_batch = [
        ["input_v0_b0", "input_v1_b0"],
        [None, "input_v1_b1"],
        [None, "input_v1_b2"],
    ]
    text_output_list_batch = [
        ["output_v0_b0", "output_v1_b0"],
        [None, "output_v1_b1"],
        [None, "output_v1_b2"],
    ]
    example_list_batch = [
        ["example_v0_b0", "example_v1_b0"],
        [None, "example_v1_b1"],
        [None, "example_v1_b2"],
    ]
    answer_list_batch = [
        ["answer_v0_b0", "answer_v1_b0"],
        [None, "answer_v1_b1"],
        [None, "answer_v1_b2"],
    ]

    result = get_output_across_videos(
        video_count=2,
        text_input_list_batch=text_input_list_batch,
        text_output_list_batch=text_output_list_batch,
        example_list_batch=example_list_batch,
        answer_list_batch=answer_list_batch,
    )

    text_outputs, text_inputs, examples, answers = result

    # Video 0: should have 1 entry (only batch 0, stops at None in batch 1)
    assert len(text_inputs[0]) == 1
    assert text_inputs[0] == ["input_v0_b0"]
    assert text_outputs[0] == ["output_v0_b0"]
    assert examples[0] == ["example_v0_b0"]
    assert answers[0] == ["answer_v0_b0"]

    # Video 1: should have 3 entries (all batches with values)
    assert len(text_inputs[1]) == 3
    assert text_inputs[1] == ["input_v1_b0", "input_v1_b1", "input_v1_b2"]
    assert text_outputs[1] == ["output_v1_b0", "output_v1_b1", "output_v1_b2"]
    assert examples[1] == ["example_v1_b0", "example_v1_b1", "example_v1_b2"]
    assert answers[1] == ["answer_v1_b0", "answer_v1_b1", "answer_v1_b2"]


def test_get_output_across_videos_misaligned_nones():
    """Test get_output_across_videos raises error when Nones are misaligned."""
    # Create test data with misaligned Nones: example is None but answer is not
    text_input_list_batch = [["input_v0", "input_v1"]]
    text_output_list_batch = [["output_v0", "output_v1"]]
    example_list_batch = [[None, "example_v1"]]  # None for video 0
    answer_list_batch = [["answer_v0", "answer_v1"]]  # Not None for video 0

    with pytest.raises(AssertionError, match="Misaligned Nones at video 0"):
        get_output_across_videos(
            video_count=2,
            text_input_list_batch=text_input_list_batch,
            text_output_list_batch=text_output_list_batch,
            example_list_batch=example_list_batch,
            answer_list_batch=answer_list_batch,
        )


def test_get_output_across_videos_none_then_value():
    """Test get_output_across_videos raises error when non-None appears after None.

    Once a video returns None, all subsequent values must be None (video has ended).
    """
    # Create test data where video 0 has None in batch 1, then a value in batch 2
    text_input_list_batch = [
        ["input_v0_b0", "input_v1_b0"],
        [None, "input_v1_b1"],
        ["input_v0_b2", "input_v1_b2"],  # Error: value after None for video 0
    ]
    text_output_list_batch = [
        ["output_v0_b0", "output_v1_b0"],
        [None, "output_v1_b1"],
        ["output_v0_b2", "output_v1_b2"],
    ]
    example_list_batch = [
        ["example_v0_b0", "example_v1_b0"],
        [None, "example_v1_b1"],
        ["example_v0_b2", "example_v1_b2"],
    ]
    answer_list_batch = [
        ["answer_v0_b0", "answer_v1_b0"],
        [None, "answer_v1_b1"],
        ["answer_v0_b2", "answer_v1_b2"],
    ]

    with pytest.raises(
        AssertionError, match="Video 0: Non-None value appeared after None"
    ):
        get_output_across_videos(
            video_count=2,
            text_input_list_batch=text_input_list_batch,
            text_output_list_batch=text_output_list_batch,
            example_list_batch=example_list_batch,
            answer_list_batch=answer_list_batch,
        )


def test_get_answer_from_completion_valid():
    """Test get_answer_from_completion with valid answer tags and percentage."""
    completion = "<think>Some reasoning here</think><answer>75%</answer>"
    result = get_answer_from_completion(completion)
    assert result == "75"


def test_get_answer_from_completion_incomplete_tag():
    """Test get_answer_from_completion with incomplete closing tag (fallback pattern)."""
    completion = "<think>Some reasoning</think><answer>-50%</answer"
    result = get_answer_from_completion(completion)
    assert result == "-50"


def test_get_answer_from_completion_clamping():
    """Test get_answer_from_completion with values that need clamping."""
    # Test upper bound clamping
    completion_over = "<answer>150%</answer>"
    result_over = get_answer_from_completion(completion_over)
    assert result_over == "100"

    # Test lower bound clamping
    completion_under = "<answer>-200%</answer>"
    result_under = get_answer_from_completion(completion_under)
    assert result_under == "-100"


def test_get_answer_from_completion_no_tags():
    """Test get_answer_from_completion when no answer tags are present."""
    completion = "<think>Just thinking, no answer provided</think>"
    result = get_answer_from_completion(completion)
    assert result == ""


def test_create_composite_frame_external_only():
    """Test create_composite_frame with external_only=True (only external views)."""
    size = 100
    # Create distinct colored frames for testing
    first_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [255, 0, 0]).astype(
        np.uint8
    )  # Red
    first_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 255, 0]).astype(
        np.uint8
    )  # Green
    frame0_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [0, 0, 255]).astype(
        np.uint8
    )  # Blue
    frame0_external = (np.ones((size, size, 3), dtype=np.uint8) * [255, 255, 0]).astype(
        np.uint8
    )  # Yellow
    frame1_wrist = (np.ones((size, size, 3), dtype=np.uint8) * [255, 0, 255]).astype(
        np.uint8
    )  # Magenta
    frame1_external = (np.ones((size, size, 3), dtype=np.uint8) * [0, 255, 255]).astype(
        np.uint8
    )  # Cyan

    composite = create_composite_frame(
        first_wrist,
        first_external,
        frame0_wrist,
        frame0_external,
        frame1_wrist,
        frame1_external,
        from_zero=False,
        external_only=True,
    )

    if VISUALIZE:
        plt.imshow(composite)
        plt.title("Composite Frame - external_only=True")
        plt.show()

    # With external_only=True and from_zero=False, we have 3 timesteps (first, frame0, frame1)
    # Each resized to 384x384, with 5px padding between columns
    # Expected width: 384 * 3 + 5 * 2 = 1162
    # Expected height: 384 (only one row - external views only, no wrist views)
    assert composite.shape == (384, 1162, 3)

    # Check that composite has 3 channels (RGB)
    assert composite.shape[2] == 3

    # Verify that only external views are present (green, yellow, cyan)
    # Check left section contains green (first_external)
    left_center = composite[192, 192]  # Center of first 384x384 block
    assert left_center[1] > 150  # Strong green component

    # Check middle section contains yellow (frame0_external)
    middle_center = composite[192, 384 + 5 + 192]
    assert middle_center[0] > 150 and middle_center[1] > 150  # Strong red and green

    # Check right section contains cyan (frame1_external)
    right_center = composite[192, 2 * (384 + 5) + 192]
    assert right_center[1] > 150 and right_center[2] > 150  # Strong green and blue


def test_prepare_vlm_images_from_zero():
    """Test prepare_vlm_images with from_zero=True (2-timestep composite mode)."""
    # Create test data with 1 episode of 3 frames
    # This should generate 2 VLM images (for t=1 and t=2)

    def create_test_image(color_rgb, size=(100, 100)):
        """Helper to create a solid color test image."""
        return (np.ones((size[0], size[1], 3), dtype=np.uint8) * color_rgb).astype(
            np.uint8
        )

    # Create a single episode with 3 frames
    episode_front = [
        create_test_image([255, 0, 0]),  # Red - t=0
        create_test_image([0, 255, 0]),  # Green - t=1
        create_test_image([0, 0, 255]),  # Blue - t=2
    ]
    episode_wrist = [
        create_test_image([255, 255, 0]),  # Yellow - t=0
        create_test_image([255, 0, 255]),  # Magenta - t=1
        create_test_image([0, 255, 255]),  # Cyan - t=2
    ]

    front_images = [episode_front]
    wrist_images = [episode_wrist]
    num_episodes = 1
    episode_lengths = [3]

    # Call prepare_vlm_images with from_zero=True
    vlm_images = prepare_vlm_images(
        front_images=front_images,
        wrist_images=wrist_images,
        num_episodes=num_episodes,
        episode_lengths=episode_lengths,
        from_zero=True,
    )

    # Verify output structure
    assert len(vlm_images) == num_episodes, "Should have 1 episode"
    assert len(vlm_images[0]) == 2, "Episode should have 2 VLM images (from t=1,2)"

    # Verify all outputs are PIL Images
    for t_idx, img in enumerate(vlm_images[0]):
        assert isinstance(img, Image.Image), f"Timestep {t_idx} should be PIL Image"
        # Verify images are not empty
        assert img.size[0] > 0 and img.size[1] > 0, "Image should have valid size"

    # Verify image dimensions after smart_resize
    for t_idx, img in enumerate(vlm_images[0]):
        width, height = img.size
        # smart_resize should produce dimensions that are multiples of factor (28)
        assert (
            width % 28 == 0
        ), f"Width {width} should be multiple of factor=28 after smart_resize"
        assert (
            height % 28 == 0
        ), f"Height {height} should be multiple of factor=28 after smart_resize"
        # With from_zero=True, composite should be square (773x773 before resize)
        # After smart_resize, aspect ratio should remain square
        assert (
            width == height
        ), f"Image should be square with from_zero=True, got {width}x{height}"

    # Optional visualization
    if VISUALIZE:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("prepare_vlm_images with from_zero=True", fontsize=16)

        for t_idx in range(2):
            axes[t_idx].imshow(vlm_images[0][t_idx])
            axes[t_idx].set_title(f"VLM Image t={t_idx+1}")
            axes[t_idx].axis("off")

        plt.tight_layout()
        plt.show()

    # Verify image content is not all zeros
    for t_idx, img in enumerate(vlm_images[0]):
        img_array = np.array(img)
        assert img_array.max() > 0, f"Timestep {t_idx} should not be all black"
