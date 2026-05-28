<h1 align="center">
  SOLE-R1: Video-Language Reasoning as the<br>
  Sole Reward for On-Robot RL
</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2603.02115-b31b1b.svg)](https://arxiv.org/abs/2603.28730)
[![Website](https://img.shields.io/badge/Website-SOLE--R1-87CEEB?logo=githubpages)](https://philip-mit.github.io/sole-r1/)
[![Model](https://img.shields.io/badge/Model-SOLE--R1--8B-blue?logo=huggingface)](https://huggingface.co/Philip-MIT/SOLE-R1-8B)
[![Data](https://img.shields.io/badge/Data-SOLE--Training-FFD21E?logo=huggingface)](https://huggingface.co/datasets/Philip-MIT/sole_training_data)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)





This is the repository for the paper:

> SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot RL
>
> Philip Schroeder, Thomas Weng, Karl Schmeckpeper, Eric Rosen, Stephen Hart, Ondrej Biza

SOLE-R1 is a video-language reasoning model designed for guiding online RL with per-timestep chain-of-thought reasoning and progress prediction.

<p align="center">
  <img src="assets/fig1_v3_padded.png"  alt="Figure 1" width="1000">
</p>

---
### Paper (arXiv)
https://arxiv.org/abs/2603.28730

### Project website and demos
https://philip-mit.github.io/sole-r1/


## Example videos comparing SOLE-R1 rewards vs Robometer, RoboReward, TOPReward, and Gemini-3-Pro

<!-- <video src="assets/robosuite_lift_episode_12_unsuccessful_max_reward_38.mp4" controls></video> -->
<!-- https://github.com/user-attachments/assets/0d804a7d-c00a-4206-98be-421c91329f8e -->

<!-- https://github.com/user-attachments/assets/4df13587-2fdf-4635-b0af-00daeed0be7e -->

<!-- https://github.com/user-attachments/assets/cd481f28-0cb3-4874-bd50-1ec3ad8326ec -->

https://github.com/user-attachments/assets/3c444096-d3dd-47c7-b09d-90b0756d0f72


## Example videos showing SOLE-R1 reasoning traces

https://github.com/user-attachments/assets/87d52be7-260c-4f0f-a9d5-8e4915bacab7

## Quick start: Example reward and reasoning generation and plotting
[RewardGen](https://github.com/Philip-MIT/rewardgen) provides the easiest way for downloading and using SOLE-R1, along with other recent reward models such as Robometer, RoboReward, and TOPReward.

```python
# install package: pip install -U rewardgen 
# or clone repo: git clone https://github.com/Philip-MIT/rewardgen

from rewardgen import generate, video_plot

video_paths = ['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_12_unsuccessful_max_reward_38.mp4']
task_description="Pick up the cube from the table."

rewards, reasoning_traces = generate(model="SOLE-R1",  task_description=task_description, video_paths=video_paths, view_type_per_video=['external and wrist'], verbose=False)
print(rewards)
# [[0.0, 4.0, 8.0, 8.666666666666666, 9.333333333333334, 10.0, 14.0, 18.0, 22.0, 28.0, 34.0, 35.0, 36.0, 37.0, 36.333333333333336, 35.666666666666664, 35.0, 33.5, 32.0, 33.0, 34.0, 35.0, 37.0, 39.0, 41.0]]

print(reasoning_traces)
# [['', ' ', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 0%, the current task progress seems to have increased to 8%.</think><answer>8%</answer>', ' ', ' ', '<think>In the previous timestep, the gripper is hovering above the table with its jaws open and is a bit offset from the red cube. In the current timestep, the arm has moved slightly closer and is better aligned over the cube, but the gripper remains open and there is still no contact or lift. Because the distance to the cube has decreased, the task is making small forward progress. Given the previous progress was 8%, this incremental approach suggests the current progress is about 10%. To complete the task, the robot still needs to finish approaching, close the gripper to secure the cube, and lift it off the table.</think><answer>10%</answer>', ' ', ' ', '<think>At the previous timestep, the gripper hovered over the table and was offset from the cube. At the current timestep, the arm has moved closer toward the cube and appears better aligned, but the gripper is still open and has not contacted or picked up the cube. Because the robot has reduced the distance to the target, progress should be increasing; given the prior 10% estimate, I would predict the current progress to be about 22%. To finish the task, the robot needs to close the remaining gap, align over the cube, grasp it securely, and lift it from the table.</think><answer>22%</answer>', ' ', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 22%, the current task progress seems to have increased to 34%.</think><answer>34%</answer>', ' ', ' ', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 34%, the current task progress seems to have increased to 37%.</think><answer>37%</answer>', ' ', ' ', '<think>At both the previous and current timestep, the cube is still sitting on the table and the gripper is not touching the cube. The current timestep shows that the gripper has moved further from the cube compared to the previous timestep. Therefore, the task progress appears to be decreasing. Given that the previous task progress was 37%, the current task progress seems to have decreased to 35%.</think><answer>35%</answer>', ' ', '<think>At the previous timestep, the cube is resting on the table and the gripper is nearby but not touching it. At the current timestep, the cube is still on the table and the gripper has shifted slightly farther from the cube, with no grasp established. This movement away from the target suggests regression rather than progress toward picking it up. Given the previous task progress of 35%, this backward motion would lead me to predict the current progress is lower, around 32%. To complete the task, the robot needs to move the gripper back toward the cube, align over it, close the fingers to grasp, and lift it from the table.</think><answer>32%</answer>', ' ', ' ', "<think>In the previous timestep, the gripper is hovering slightly offset from the cube on the table with a small gap remaining. In the current timestep, the arm has advanced and rotated so the fingers are nearer to the cube's side, but there is still no contact or lift. This closer approach suggests incremental progress; given the previous progress of 32%, I would estimate the current progress to be about 35%. To complete the task, the robot still needs to make contact, close the gripper to secure the cube, and lift it off the table.</think><answer>35%</answer>", ' ', ' ', '<think>At the previous timestep, the gripper is hovering near the red cube with a small gap, not yet in contact. At the current timestep, the gripper has moved closer and appears to be touching the side of the cube, indicating the approach is improving and alignment is better. Because contact has been established compared to the previous timestep, the task progress should increase from the prior 35%, and a reasonable prediction for the current timestep is about 41%. To complete the task, the robot still needs to secure a grasp on the cube, lift it fully off the table, and stabilize it in the gripper.</think><answer>41%</answer>']]


## Plotting with show_reasoning_traces=True
output_sole = {"model": "SOLE-R1", "rewards": rewards[0], "reasoning_traces": reasoning_traces[0]}

video_plot(
    outputs=[output_sole], 
    plot_save_path='model_outputs/sole-r1/robosuite/lift/unsuccessful/robosuite_lift_episode_12_unsuccessful_max_reward_38.mp4', 
    video_path=video_paths[0],
    show_reasoning_traces=True,
    task_description=task_description,
    verbose=False
)
```

## Reward generation and plotting across many videos

```python
from rewardgen import generate, video_plot

import glob
video_paths = glob.glob('test_videos/robosuite/lift/unsuccessful/*.mp4')
# video_paths = glob.glob('test_videos/robosuite/lift/successful/*.mp4')
task_description="Pick up the cube from the table."


## REWARD GENERATION
rewards, reasoning_traces = generate(model="SOLE-R1",  task_description=task_description, video_paths=video_paths, view_type='external and wrist', verbose=False)

## PLOTTING
import json
plot_save_dir = 'model_outputs/sole-r1/'
for video_idx in range(len(video_paths)):
    output_sole = {"model": "SOLE-R1", "rewards": rewards[video_idx], "reasoning_traces": reasoning_traces[video_idx]}
    # Optional: Ground-truth rewards (available for test videos from sim environments)
    with open(video_paths[video_idx].replace(".mp4", "/data.json"), 'r') as f:
        data = json.load(f)
    #
    output_groundtruth = {"model": "Ground truth", "rewards": data['ground-truth rewards']}
    video_plot(
        outputs = [output_groundtruth, output_sole], 
        plot_save_path = plot_save_dir + video_paths[video_idx].split('test_videos/')[-1] , 
        video_path = video_paths[video_idx],
        task_description=task_description,
        verbose = False
    )
```



<!-- ## 🎥 Demos

Example videos demonstrating SOLE-R1 frame-level reasoning and task progress prediction can be found at:

https://philip-mit.github.io/sole-r1/

--- -->

## Model Checkpoints

Final model checkpoint available in HF format at [SOLE-R1-8B](https://huggingface.co/Philip-MIT/SOLE-R1-8B) 

```python
# Optional: pre-download model checkpoint using RewardGen 
from rewardgen.utils.model_utils import get_model_dir
get_model_dir('sole-r1')
```


---

## Training Dataset

Full training dataset (2TB) is available in HF format at [sole_training_data](https://huggingface.co/datasets/Philip-MIT/sole_training_data) 

### Streaming
```python
from datasets import load_dataset

ds = load_dataset(
    "Philip-MIT/sole_training_data",
    split="train",
    streaming=True,
)

# Print one example
for row in ds:
    print(row)
    break
    # {
    #     'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1176x784 at 0x7F5F26A107C0>, 
    #     'prompt': '[{"role": "system", "content": "You are an expert roboticist with the goal of predicting task progress percentages given frames from a video of a robot attempting to complete a task. You first think, in the form of an internal monologue, before providing your final answer. Your reasoning process MUST BE enclosed within <think> </think> tags and should include detailed reasoning. Your final answer MUST BE enclosed within <answer> </answer> tags and should be a integer (positive or negative) representing current task progress percentage. Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Here is an image containing multiple camera views of a robot attempting to complete a task. The views on the top are from an external camera. The views on the bottom are from the robot\'s wrist camera. The views from the very first timestep are shown to the left. The views from the previous timestep are shown in the middle. The views from the current timestep are shown to the right. The task description is: close the left drawer. The task progress for the very first timestep is 0%. The task progress for the previous timestep is 0%. Predict the task progress for the current timestep."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>At the previous timestep, the left drawer is open and the gripper is off to the side with no contact to the drawer or its handle. At the current timestep, the gripper has shifted slightly closer to the face of the left drawer while still not making contact, and the drawer remains open. This small approach suggests initial progress toward closing the drawer. Given the previous task progress of 0%, I would predict the current task progress to be about 1%. To complete the task, the robot still needs to reach the drawer front or handle, establish contact, and push the drawer fully shut.</think><answer>1%</answer>"}]}]', 
    #     'solution': '<think>At the previous timestep, the left drawer is open and the gripper is off to the side with no contact to the drawer or its handle. At the current timestep, the gripper has shifted slightly closer to the face of the left drawer while still not making contact, and the drawer remains open. This small approach suggests initial progress toward closing the drawer. Given the previous task progress of 0%, I would predict the current task progress to be about 1%. To complete the task, the robot still needs to reach the drawer front or handle, establish contact, and push the drawer fully shut.</think><answer>1%</answer>', 
    #     'image_name': 'CloseDrawer_demo_526_lev7-two-two_timestep_0_8.jpg', 
    #     'from_annotation': 1, 
    #     'completion': '<think>At the previous timestep, the left drawer is open and the gripper is off to the side with no contact to the drawer or its handle. At the current timestep, the gripper has shifted slightly closer to the face of the left drawer while still not making contact, and the drawer remains open. This small approach suggests initial progress toward closing the drawer. Given the previous task progress of 0%, I would predict the current task progress to be about 1%. To complete the task, the robot still needs to reach the drawer front or handle, establish contact, and push the drawer fully shut.</think><answer>1%</answer>', 
    #     'data_source': 'processed_0717_annot2_CloseDrawer_sft_hf'
    # }
```


### Dowloading the full dataset (2TB) to a local directory
```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="Philip-MIT/sole_training_data",
    repo_type="dataset",
    local_dir="/path/to/local/sole_training_data",
)
```
---

## Optional: Download all test videos and example model outputs 
```bash
# 1) Install gcloud: https://cloud.google.com/sdk/docs/install

# 2) Go to target directory
# cd /path/to/rewardgen

# Optional: disable credentials so you don't have to authenticate
gcloud config set auth/disable_credentials True

# Download test videos
gcloud storage cp --recursive gs://roboreason-view-videos-philip/test_videos ./

# Download model outputs for all test videos (including outputs from SOLE-R1, Robometer, RoboReward, TOPReward, and Gemini-3-Pro)
gcloud storage cp --recursive gs://roboreason-view-videos-philip/model_outputs ./

# Optional: re-enable credentials afterward if you disabled them above.
gcloud config set auth/disable_credentials False

```



## Reward Server and Reward Client

`reward_server/` runs a reward inference service for SOLE-R1. Given a task string plus robot video frames, it uses a local SOLE-R1 checkpoint to predict dense per-timestep task-progress rewards and returns both parsed progress scores and raw model reasoning outputs.

```bash
cd reward_server
bash run.sh --checkpoint-path /path/to/sole-r1-checkpoint --port 8001
```

The default service name is `rewards`. Inputs include `task`, `front_images`, optional `wrist_images`, `temperature`, `from_zero`, and `external_only`. The server can score either front-only video or paired front/wrist views, comparing frames against the start state and, by default, the previous timestep.


`reward_client/` is a CLI for querying a running reward server from local robot videos, image folders, NumPy arrays, or HDF5 files. It samples frames, sends them to the server, prints per-timestep progress rewards, and can optionally save JSON results and an annotated visualization video.

```bash
cd reward_client
uv sync

uv run python main.py \
  --host localhost \
  --port 8001 \
  --service-name rewards \
  --front /path/to/front_video.webm \
  --wrist /path/to/wrist_video.webm \
  --task "open the drawer" \
  --output-file rewards.json \
  --video-output rewards.webm
```

Omit `--wrist` for front-camera-only scoring. Use `--from-zero` to score each timestep independently relative to the first frame, or use `--target-fps`, `--source-fps`, and `--stride` to control frame sampling.

---

## Citation

If you use SOLE-R1 data or models in your research, please cite:

    @article{schroeder2026soler1,
        title         = {SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning},
        author        = {Schroeder, Philip and Weng, Thomas and Schmeckpeper, Karl and Rosen, Eric and Hart, Stephen and Biza, Ondrej},
        journal       = {arXiv preprint arXiv:2603.28730},
        year          = {2026},
        eprint        = {2603.28730},
        archivePrefix = {arXiv},
        primaryClass  = {cs.RO},
        doi           = {10.48550/arXiv.2603.28730},
        url           = {https://arxiv.org/abs/2603.28730}
    }


## License

This project is released under the MIT License unless otherwise specified.  
<!-- See the LICENSE file for details. -->

---

## Contact

For questions or issues, please open a GitHub Issue or contact the authors directly.

---


