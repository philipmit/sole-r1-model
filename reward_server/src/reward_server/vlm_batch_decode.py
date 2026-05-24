# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy as cp
import gc
import json
import os
from pathlib import Path

import PIL
import torch
from accelerate.utils import gather_object
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from trl.data_utils import maybe_apply_chat_template
from vllm import LLM, SamplingParams

from reward_server.utils import (
    assemble_output_batch,
    get_answer_from_completion,
    make_conversation_image,
)


def load_model(
    checkpoint_path: Path,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    max_model_len: int = 131072,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 200,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
):
    """Load a Qwen VLM model and preprocessor.

    Args:
        checkpoint_path (Path): Path to the model checkpoint.
        gpu_memory_utilization (float): GPU memory utilization for loading the model.
        enable_prefix_caching (bool): Whether to enable prefix caching.
        max_model_len (int): Maximum sequence length.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.
        top_k (int): Top-k sampling parameter.
        max_tokens (int): Maximum number of tokens to generate.
        min_pixels (int): Minimum number of pixels for image processing.
        max_pixels (int): Maximum number of pixels for image processing.

    There are two quirks to resolve.
    1. This function first edits the saved model files, then loads and saved the model
       and then loads it again.
    2. We probably don't have to return two preprocessors.

    Returns:
        llm: Loaded LLM model.
        processing_class: Preprocessor object with updated parameters.
        processor: Original preprocessor object.
        sampling_params: Default sampling parameters for generation.
    """

    # Load only Qwen 3 8B.
    assert "q3vl8b", "We're only supporting Qwen 3 8B for these experiments."
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    # Load and save checkpoint with new preprocessor config.
    os.system(f"cp data/preprocessor_config.json {checkpoint_path}/")
    processor = AutoProcessor.from_pretrained(checkpoint_path, fix_mistral_regex=True)
    os.system(f"rm {checkpoint_path}/preprocessor_config.json")
    processor.save_pretrained(f"{checkpoint_path}")
    processor = None

    # Load model and processor.
    processing_class = AutoProcessor.from_pretrained(model_id)
    pad_token_id = processing_class.tokenizer.pad_token_id
    processing_class.pad_token_id = pad_token_id
    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
    processing_class.image_processor.max_pixels = max_pixels
    processing_class.image_processor.min_pixels = min_pixels

    processor = AutoProcessor.from_pretrained(checkpoint_path)

    llm = LLM(
        model=checkpoint_path,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    return llm, processing_class, processor, sampling_params


def vlm_batch_decode(
    processing_class,
    processor,
    llm,
    sampling_params,
    video_count: int,
    video_step_count: list[int],
    vlm_images: list[list[Image.Image]],
    question: str,
    max_prompt_length: int = 2048,
    from_zero: bool = False,
):
    """Batch decode VLM model across multiple videos and time steps.

    Args:
        processing_class: Preprocessor object for preparing inputs.
        processor: Original preprocessor object.
        llm: Loaded LLM model.
        sampling_params: Sampling parameters for generation.
        video_count: Number of videos to process.
        video_step_count: List of step counts for each video.
        vlm_images: List of lists of images for each video.
        question: Question prompt to use for each time step.
        max_prompt_length: Maximum length of the prompt input.
        from_zero: If False, includes previous progress in the prompt.
    """
    max_video_step = max(video_step_count)

    # batch decoding
    text_output_list_batch = []
    text_input_list_batch = []
    example_list_batch = []
    answer_list_batch = []
    prev_answer_batch = [["0" for _ in range(video_count)]]

    # Subtract one because we don't predict progress for the first time step.
    for video_step in range(max_video_step - 1):
        gc.collect()

        current_batch_input = []
        current_indices = []
        current_examples = []

        for ep_idx in range(video_count):
            # Subtract one because we don't predict progress for the first time step.
            # We video_step_count == 3, then we have 2 images and 2 predictions to make.
            if video_step >= video_step_count[ep_idx] - 1:
                continue
            else:
                image = vlm_images[ep_idx][video_step]
                tmp_example = {
                    "image": image,
                    "question": cp.deepcopy(question),
                }
                current_examples.append(tmp_example)

                prompt = make_conversation_image(tmp_example["question"])

                inputs = [
                    {
                        "image": image,
                        "problem": tmp_example["question"],
                        "image_name": "hello123",
                        "prompt": prompt,
                    }
                ]

                prompts = [x["prompt"] for x in inputs]

                prompts_text = [
                    maybe_apply_chat_template(example, processing_class)["prompt"]
                    for example in inputs
                ]

                images = [x["image"] for x in inputs]

                prompt_inputs = processing_class(
                    text=prompts_text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )

                batch_size = 1
                batched_inputs = {
                    k: v.repeat(batch_size, *[1] * (v.dim() - 1))
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in prompt_inputs.items()
                }

                if max_prompt_length is not None:
                    batched_inputs["input_ids"] = batched_inputs["input_ids"][
                        :, -max_prompt_length:
                    ]
                    batched_inputs["attention_mask"] = batched_inputs["attention_mask"][
                        :, -max_prompt_length:
                    ]

                inputs_vllm = []
                for image_data, messages in zip(images, prompts):
                    prompt = processing_class.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_data, _ = (
                        process_vision_info(messages)
                        if not isinstance(image_data, PIL.Image.Image)
                        else (image_data, None)
                    )
                    for i in range(batch_size):
                        inputs_vllm.append(
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": image_data},
                            }
                        )

                all_inputs_vllm = gather_object(inputs_vllm)

                if not from_zero:
                    prev_answer = prev_answer_batch[video_step][ep_idx]
                    replace_start = "The task progress for the previous timestep is "
                    replace_end = "%. "
                    all_inputs_vllm[0]["prompt"] = (
                        all_inputs_vllm[0]["prompt"].split(replace_start)[0]
                        + replace_start
                        + prev_answer
                        + replace_end
                        + all_inputs_vllm[0]["prompt"]
                        .split(replace_start)[1]
                        .split(replace_end)[1]
                    )

                current_batch_input.append(all_inputs_vllm[0])
                current_indices.append(ep_idx)

        retry_gen_count = 0
        while retry_gen_count < 3:
            outputs = None
            del outputs
            outputs = llm.generate(
                current_batch_input,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if not len(outputs) == len(current_batch_input):
                print(retry_gen_count)
                print(
                    f"Outputs length {len(outputs)} is not equal to inputs length {len(current_batch_input)}"
                )
                retry_gen_count = retry_gen_count + 1
                # assert False, f"Outputs length {len(outputs)} is not equal to inputs length {len(current_video_idx_batch_input)}"
            else:
                retry_gen_count = 999

        completion_ids = [
            out.token_ids for completions in outputs for out in completions.outputs
        ]

        text_output = processor.batch_decode(completion_ids, skip_special_tokens=True)

        text_output_list_batch = text_output_list_batch + [
            assemble_output_batch(text_output, current_indices, video_count)
        ]
        example_list_batch = example_list_batch + [
            assemble_output_batch(current_examples, current_indices, video_count)
        ]

        text_input_list_batch = text_input_list_batch + [
            assemble_output_batch(
                [
                    current_video_idx_batch_input_i["prompt"]
                    for current_video_idx_batch_input_i in current_batch_input
                ],
                current_indices,
                video_count,
            )
        ]
        prev_answer_batch = prev_answer_batch + [
            assemble_output_batch(
                [
                    get_answer_from_completion(text_output_i)
                    for text_output_i in text_output
                ],
                current_indices,
                video_count,
            )
        ]
        answer_list_batch = answer_list_batch + [
            assemble_output_batch(
                [
                    get_answer_from_completion(text_output_i)
                    for text_output_i in text_output
                ],
                current_indices,
                video_count,
            )
        ]

    return (
        text_input_list_batch,
        text_output_list_batch,
        example_list_batch,
        answer_list_batch,
    )
