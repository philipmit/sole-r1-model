# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import google.genai as genai
import hydra
import numpy as np
from google.genai.types import ThinkingLevel
from omegaconf import DictConfig
from PIL import Image

from reward_server.constants import (
    API_RESPONSE_REPLACEMENTS,
    SYSTEM_PROMPT_GEMINI,
    USER_PROMPT_TEMPLATE_GEMINI,
)
from reward_server.utils import (
    InferenceServer,
    count_images,
    process_images,
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3-pro-preview"


def gemini(
    model: genai.Client,
    task_description_i: str,
    frame_list: list[Image.Image],
    try_count_max: int = 3,
    verbose: bool = True,
) -> tuple[list[int], list[str], list]:
    """Predict task progress for a single episode using Gemini.

    Args:
        model: The Gemini model client.
        task_description_i: Description of the task.
        frame_list: List of PIL Image frames.
        try_count_max: Maximum number of retry attempts for API calls.
        verbose: Whether to print detailed logs.

    Returns:
        progress_list: List of predicted progress percentages.
        response_text_list: List of raw response texts from the model.
        prompt_list: List of prompts sent to the model.
    """
    current_progress = 0
    prompt_list = []
    progress_list = []
    response_text_list = []
    messages_content = []
    response_text = None
    for current_idx in range(1, len(frame_list)):
        try_count = 0
        while try_count < try_count_max:
            try:
                messages_content = [
                    USER_PROMPT_TEMPLATE_GEMINI.format(
                        task_description=task_description_i,
                        prev_progress=current_progress,
                    ),
                    Image.fromarray(frame_list[current_idx - 1]),
                    Image.fromarray(frame_list[current_idx]),
                ]
                response = model.models.generate_content(
                    model=MODEL_NAME,
                    contents=messages_content,
                    config={
                        "system_instruction": SYSTEM_PROMPT_GEMINI,
                        "max_output_tokens": 6000,
                        "temperature": 0,
                        "top_k": 1,
                        "thinking_config": {
                            "include_thoughts": True,
                            "thinking_level": ThinkingLevel.LOW,
                        },
                    },
                )

                response_text = response.candidates[0].content.parts[-1].text

                for src, target in API_RESPONSE_REPLACEMENTS.items():
                    response_text = response_text.replace(src, target)

                current_progress = int(
                    response_text.split("<answer>")[1]
                    .split("</answer>")[0]
                    .replace("%", "")
                    .strip()
                )
                progress_list.append(current_progress)
                response_text_list.append(response_text)
                prompt_list.append(messages_content[0])
                break
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
                try_count += 1
                if "quota" in str(e).lower():
                    print("Quota exceeded, sleeping for 60 seconds.")
                    time.sleep(60)
                print(f"Response: {response_text}")

        if try_count >= try_count_max:
            response_text_list.append("")
            if len(progress_list) > 0:
                current_progress = progress_list[-1]
            else:
                current_progress = 0
            progress_list.append(current_progress)
            if len(messages_content) > 0:
                prompt_list.append(messages_content[0])
            else:
                prompt_list.append("")

        if verbose:
            print(
                "\n\n*******************************************************************************"
            )
            print(prompt_list[-1])
            print("\n----------------- Response -----------------")
            print(response_text_list[-1])
            print(progress_list[-1])

    return progress_list, response_text_list, prompt_list


@hydra.main(version_base=None, config_path="config", config_name="gemini")
def main(cfg: DictConfig) -> None:
    google_model = genai.Client(api_key=API_KEY)

    receiver = InferenceServer(
        port_num=cfg.server_port,
    )

    def callback(payload: dict[str, Any]) -> np.ndarray:
        if "front_images" not in payload:
            return {"success": False, "message": "No 'front_images' in payload."}
        if "task" not in payload:
            return {"success": False, "message": "No 'task' in payload."}

        try:
            front_images = payload["front_images"]
            task = payload["task"]

            num_episodes, _ = count_images(front_images)
            processed_front_images = process_images(front_images)

            answer_list_list = []
            response_text_list_list = []
            prompt_list_list = []

            def process_episode(episode_idx):
                progress_list, response_text_list, prompt_list = gemini(
                    google_model,
                    task,
                    processed_front_images[episode_idx],
                    verbose=cfg.verbose,
                )
                if cfg.verbose:
                    print("\n\n----------------------------------------")
                    print(f"Episode {episode_idx}.")
                    print(f"progress_list: {progress_list}")
                return episode_idx, progress_list, response_text_list, prompt_list

            with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
                futures = [
                    executor.submit(process_episode, episode_idx)
                    for episode_idx in range(num_episodes)
                ]
                results = [future.result() for future in futures]

            # Sort results by episode_idx to maintain order
            results.sort(key=lambda x: x[0])

            for _, progress_list, response_text_list, prompt_list in results:
                answer_list_list.append([str(v) for v in progress_list])
                response_text_list_list.append(response_text_list)
                prompt_list_list.append(prompt_list)

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
                    "text_outputs": response_text_list_list,
                    "text_inputs": prompt_list_list,
                },
            }

        except Exception as e:
            return {"success": False, "message": str(e) + traceback.format_exc()}

    receiver.register_interface(cfg.service_name, callback)
    receiver.start(threaded=False)


if __name__ == "__main__":
    main()
