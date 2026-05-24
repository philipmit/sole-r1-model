# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert roboticist with the goal of predicting task progress percentages given frames from a video of a robot attempting to complete a task. "
    + "You first think, in the form of an internal monologue, before providing your final answer. "
    + "Your reasoning process MUST BE enclosed within <think> </think> tags and should include detailed reasoning. "
    + "Your final answer MUST BE enclosed within <answer> </answer> tags and should be a integer (positive or negative) representing current task progress percentage. "
    + "Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)
USER_QUESTION_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    + "The views on the top are from an external camera. The views on the bottom are from the robot's wrist camera. "
    + "The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. "
    + "The task description is: {task_description}. "
    + "The task progress for the very first timestep is 0%. The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep."
)
USER_QUESTION_FROM_ZERO_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    + "The views on the top are from an external camera. The views on the bottom are from the robot's wrist camera. "
    + "The views from the very first timestep are shown to the left. The views from the current timestep are shown to the right. "
    + "The task description is: {task_description}. "
    + "The task progress for the very first timestep is 0%. Predict the task progress for the current timestep."
)
USER_QUESTION_EXTERNAL_VIEW_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    + "The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. "
    + "The task description is: {task_description}. "
    + "The task progress for the very first timestep is 0%. The task progress for the previous timestep is {prev_progress}%. Predict the task progress for the current timestep."
)
USER_QUESTION_EXTERNAL_VIEW_FROM_ZERO_TEMPLATE = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    + "The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. "
    + "The task description is: {task_description}. "
    + "The task progress for the very first timestep is 0%. Predict the task progress for the current timestep."
)

USER_PROMPT_TEMPLATE_GEMINI = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    + "The first image is from the previous timestep. The second image is from the current timestep. "
    + "The task description is: {task_description}. "
    + "The predicted task progress for the previous timestep was {prev_progress}%. Predict the task progress for the current timestep. "
    + "Note that the previous progress value might not be accurate. Please carefully assess the images to determine the correct progress for the current timestep. "
    + "Also note that the performance of the robot is unknown so progress can increase or decrease at any timestep. "
    + "Before providing your final answer, first briefly provide a few words that reason about what is happening at the current timestep relative to the previous timestep. "
    + "Your reasoning process should be no more than one or two sentences and MUST BE enclosed within <think> </think> tags. IMPORTANT: Do not refer to 'the user' in your reasoning. Only reason as though it is an internal monologue thinking about the robot. If you are unsure about the progress, please try to avoid overestimation and provide a conservative estimate. Please refer to the images as timesteps instead of as separate images. "
    + "Your final answer MUST BE enclosed within <answer> </answer> tags and should be an integer representing current task progress percentage. "
    + "Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)
USER_PROMPT_TEMPLATE_GPT = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    + "The first image is from the previous timestep. The second image is from the current timestep. "
    + "The task description is: {task_description}. "
    + "The predicted task progress for the previous timestep was {prev_progress}%. Predict the task progress for the current timestep. "
    + "Before providing your final answer, first briefly provide a few sentences that reason about what is happening at the current timestep relative to the previous timestep. "
    + "Your reasoning process should be no more than a few sentences and MUST BE enclosed within <think> </think> tags. Please refer to the images as timesteps instead of as separate images. "
    + "Your final answer MUST BE enclosed within <answer> </answer> tags and should be an integer representing current task progress percentage. "
    + "Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)

QUESTION_TEMPLATE = "{question}"
PROBLEM_KEY = "question"
ANSWER_KEY = "answer"
IMAGE_KEY = "image"

API_RESPONSE_REPLACEMENTS = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
}

SYSTEM_PROMPT_GEMINI = (
    "You are an expert roboticist with the goal of predicting task progress percentages given frames from a video of a robot attempting to complete a task."
)
