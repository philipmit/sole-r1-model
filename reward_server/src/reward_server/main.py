# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import traceback
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
from vllm import SamplingParams

from reward_server.constants import (
    USER_QUESTION_EXTERNAL_VIEW_FROM_ZERO_TEMPLATE,
    USER_QUESTION_EXTERNAL_VIEW_TEMPLATE,
    USER_QUESTION_FROM_ZERO_TEMPLATE,
    USER_QUESTION_TEMPLATE,
)
from reward_server.utils import (
    InferenceServer,
    count_and_validate_images,
    get_output_across_videos,
    prepare_vlm_images,
    process_images,
)
from reward_server.vlm_batch_decode import (
    load_model,
    vlm_batch_decode,
)


def callback(
    payload: dict[str, Any],
    llm,
    processing_class,
    processor,
    max_prompt_length,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 200,
) -> np.ndarray:
    """Callback for annotating rewards for a batch of episodes.

    Args:
        payload (dict): A dictionary with keys:
            - "task": str, the task name
            - "front_images": a list of episodes, each episode is a list of front images.
                Images can be formatted as numpy or torch arrays (uint8 or float32) or
                as a list or an array of compressed bytes (e.g., PNG or JPEG).
            - "wrist_images": a list of episodes, each episode is a list of wrist images.
                Save formatting as above.
            - "temperature" (optional): float, temperature for sampling. Defaults to 1.0.
            - "from_zero" (optional): bool, whether to predict progress from zero at each time step.
                Defaults to False.
            - "external_only" (optional): bool, whether to use only external camera views.
        llm: Loaded LLM model.
        processing_class: Preprocessor object for preparing inputs.
        processor: Original preprocessor object.
        sampling_params: Sampling parameters for generation.
        max_prompt_length: Maximum length of the prompt input.
        top_p: float, top-p sampling parameter.
        top_k: int, top-k sampling parameter.
        max_tokens: int, maximum number of tokens to generate.

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
    """
    if "front_images" not in payload:
        return {"success": False, "message": "No 'front_images' in payload."}
    if "wrist_images" not in payload:
        return {"success": False, "message": "No 'wrist_images' in payload."}
    if "task" not in payload:
        return {"success": False, "message": "No 'task' in payload."}

    try:
        front_images = payload["front_images"]
        wrist_images = payload["wrist_images"]
        task = payload["task"]
        temp = payload.get("temperature", 1.0)
        from_zero = payload.get("from_zero", False)
        external_only = payload.get("external_only", False)

        new_sampling_params = SamplingParams(
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        num_episodes, episode_lengths = count_and_validate_images(
            front_images=front_images,
            wrist_images=wrist_images,
        )

        processed_front_images = process_images(front_images)
        processed_wrist_images = process_images(wrist_images)

        vlm_images = prepare_vlm_images(
            front_images=processed_front_images,
            wrist_images=processed_wrist_images,
            num_episodes=num_episodes,
            episode_lengths=episode_lengths,
            from_zero=from_zero,
            external_only=external_only,
        )

        if from_zero:
            if external_only:
                question = USER_QUESTION_EXTERNAL_VIEW_FROM_ZERO_TEMPLATE.format(
                    task_description=task,
                )
            else:
                question = USER_QUESTION_FROM_ZERO_TEMPLATE.format(
                    task_description=task,
                )
        else:
            if external_only:
                question = USER_QUESTION_EXTERNAL_VIEW_TEMPLATE.format(
                    task_description=task,
                    prev_progress=0,
                )
            else:
                question = USER_QUESTION_TEMPLATE.format(
                    task_description=task,
                    prev_progress=0,
                )

        (
            text_input_list_batch,
            text_output_list_batch,
            example_list_batch,
            answer_list_batch,
        ) = vlm_batch_decode(
            processing_class=processing_class,
            processor=processor,
            llm=llm,
            sampling_params=new_sampling_params,
            video_count=num_episodes,
            video_step_count=episode_lengths,
            vlm_images=vlm_images,
            question=question,
            max_prompt_length=max_prompt_length,
            from_zero=from_zero,
        )

        (
            text_output_list_list,
            text_input_list_list,
            example_list_list,
            answer_list_list,
        ) = get_output_across_videos(
            num_episodes,
            text_input_list_batch,
            text_output_list_batch,
            example_list_batch,
            answer_list_batch,
        )

        valid_answers = []
        for episode in answer_list_list:
            valid_answers_ = [0]  # Assume the first step is always 0.
            for ans in episode:
                try:
                    progress = int(ans)
                except (ValueError, TypeError):
                    progress = valid_answers_[-1]
                valid_answers_.append(progress)
            valid_answers.append(np.array(valid_answers_, dtype=np.int32))

        return {
            "success": True,
            "data": {
                "valid_answers": valid_answers,
                "text_outputs": text_output_list_list,
                "text_inputs": text_input_list_list,
            },
        }
    except Exception as e:
        return {"success": False, "message": str(e) + traceback.format_exc()}


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    # Load model using config
    checkpoint_path = Path(cfg.checkpoint_path).expanduser().resolve()
    print("**************************************************")
    print(f"cfg.checkpoint_path: {cfg.checkpoint_path}")
    print(f"checkpoint_path: {checkpoint_path}")
    print("**************************************************")
    llm, processing_class, processor, sampling_params = load_model(
        checkpoint_path=str(checkpoint_path),
        **cfg.get("load_model_params", {}),
    )

    callback_ = partial(
        callback,
        llm=llm,
        processing_class=processing_class,
        processor=processor,
        max_prompt_length=cfg["max_prompt_length"],
        top_p=cfg.get("load_model_params", {}).get("top_p", 0.9),
        top_k=cfg.get("load_model_params", {}).get("top_k", 50),
        max_tokens=cfg.get("load_model_params", {}).get("max_tokens", 200),
    )

    receiver = InferenceServer(
        port_num=cfg.server_port,
    )

    receiver.register_interface(cfg.service_name, callback_)
    receiver.start(threaded=False)


if __name__ == "__main__":
    main()
