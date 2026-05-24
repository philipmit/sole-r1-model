import logging
import time
from typing import Set

import cv2
import matplotlib
import numpy as np
from agentlace.zmq_wrapper.req_rep import ReqRepClient
from PIL import Image

matplotlib.use("Agg")  # Set non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class InferenceClient:
    def __init__(
        self,
        server_ip: str,
        port_num: int,
        log_level=logging.INFO,
        timeout_ms: int = 1 * 3600 * 1000,  # 1 hour timeout :)
        wait_for_server: bool = False,
    ):
        """Create a ZMQ client connected to the reward inference server.

        Args:
            server_ip: IP address or hostname of the reward server.
            port_num: ZMQ port number the server is listening on.
            log_level: Logging level for the underlying ZMQ client.
            timeout_ms: Request timeout in milliseconds. Defaults to 1 hour.
            wait_for_server: If True, block until the server responds to a
                list_interfaces probe before returning.
        """
        self.client = ReqRepClient(
            server_ip, port_num, log_level=log_level, timeout_ms=timeout_ms
        )

        if wait_for_server:
            self.wait_for_server()

    def interfaces(self) -> Set[str]:
        """
        Returns the set of interfaces available on the server
            :return: Set of interfaces available on the server
        """
        response = self.client.send_msg({"type": "list_interfaces"})
        if response:
            return set(response.get("interfaces", []))
        return set()

    def call(self, name: str, payload: dict) -> dict | None:
        """
        Calls the interface on the server with the given payload
            :param name: Name of the interface
            :param payload: Payload to send to the interface
            :return: Response from the interface
        """
        return self.client.send_msg(
            {"type": "call_interface", "interface": name, "payload": payload}
        )

    def wait_for_server(self) -> None:
        res = self.interfaces()

        while len(res) == 0:
            logging.warning("Failed to connect to inference server, retrying...")
            time.sleep(30)
            res = self.interfaces()


def create_video_with_plot(
    output_video_path,
    frame_list,
    frame_desription_list,
    data_points,
    data_points_env=None,
    fps_=2,
    wrap_width=26,
    font_scale=0.5,
):
    """
    Creates a video combining robot frames, a text description panel, and a
    live-updating reward plot side by side.

    Args:
        output_video_path: Path to the output video file (.mp4 or .webm).
        frame_list: List of frames (PIL Images or numpy arrays) per step.
        frame_desription_list: List of text descriptions for each frame.
        data_points: List of predicted task progress values (0-100) per step.
        data_points_env: Optional list of environment reward values per step.
        fps_: Frames per second for the output video.
        wrap_width: Character width for text wrapping in the description panel.
        font_scale: Font scale for the description text overlay.
    """
    plt.rcParams.update({"font.size": 6})

    first_frame = frame_list[0]
    if isinstance(first_frame, Image.Image):
        first_frame = np.array(first_frame.convert("RGB"))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

    frame_width = first_frame.shape[1]
    frame_height = first_frame.shape[0]
    if frame_height < 384:
        first_frame = cv2.copyMakeBorder(
            first_frame,
            (384 - frame_height) // 2,
            (384 - frame_height) // 2,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    if frame_width < 384:
        first_frame = cv2.copyMakeBorder(
            first_frame,
            0,
            0,
            (384 - frame_width) // 2,
            (384 - frame_width) // 2,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    frame_width = first_frame.shape[1]
    frame_height = first_frame.shape[0]

    if frame_width == 2 * frame_height:
        output_width = int(2 * frame_width)
        denom_ = 2
    else:
        output_width = int(3 * frame_width)
        denom_ = 1

    output_height = frame_height

    if output_video_path.endswith(".webm"):
        fourcc = cv2.VideoWriter_fourcc("V", "P", "9", "0")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps_, (output_width, output_height)
    )

    fig, ax = plt.subplots(dpi=200)
    canvas = FigureCanvas(fig)
    xdata, ydata, ydata2 = [], [], []
    ax.set_xlim(0, len(data_points) - 1)
    ax.set_title("")
    env_points = data_points_env if data_points_env is not None else []
    ax.set_ylim(
        min(data_points + env_points),
        max([100] + data_points + env_points),
    )
    ax.set_ylabel("Task progress (%)")
    ax.set_xlabel("Step number")

    (ln,) = ax.plot([], [], "r", label="Predicted task progress")
    ln.set_color("darkblue")
    ln.set_markerfacecolor("darkblue")
    ln.set_markersize(5)

    if data_points_env is not None:
        (ln2,) = ax.plot([], [], "b", label="Environment reward")
        ln2.set_color("darkred")
        ln2.set_markerfacecolor("darkred")
        ln2.set_markersize(5)
    else:
        ln2 = None

    ax.legend(fontsize="x-small")

    for frame_number in range(len(data_points)):
        frame = frame_list[frame_number]
        if isinstance(frame, Image.Image):
            frame = np.array(frame.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        if frame_height < 384:
            frame = cv2.copyMakeBorder(
                frame,
                (384 - frame_height) // 2,
                (384 - frame_height) // 2,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
        if frame_width < 384:
            frame = cv2.copyMakeBorder(
                frame,
                0,
                0,
                (384 - frame_width) // 2,
                (384 - frame_width) // 2,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        fig.set_size_inches((frame_width // denom_) / fig.dpi, frame_height / fig.dpi)
        fig.tight_layout(pad=0.4)

        xdata.append(frame_number)
        ydata.append(data_points[frame_number])
        ln.set_data(xdata, ydata)
        if ln2 is not None:
            ydata2.append(data_points_env[frame_number])
            ln2.set_data(xdata, ydata2)
        ax.draw_artist(ax.patch)
        ax.draw_artist(ln)
        if ln2 is not None:
            ax.draw_artist(ln2)
        canvas.draw()
        plot_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
        plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,))

        plot_image_resized = cv2.resize(
            plot_image,
            (frame_width // denom_, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )
        plot_image_rgb = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2RGB)

        white_box_width = frame_width // denom_
        white_box = np.ones((frame_height, white_box_width, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        thickness = 1
        line_type = 2

        text = frame_desription_list[frame_number]
        language_description_split = text.split(" ")
        language_description_wrapped = ""
        line_i_len = 0
        for i in range(len(language_description_split)):
            if line_i_len < wrap_width:
                language_description_wrapped += language_description_split[i] + " "
                line_i_len += len(language_description_split[i])
            else:
                language_description_wrapped += "\n" + language_description_split[i] + " "
                line_i_len = len(language_description_split[i])
        text = language_description_wrapped
        text = text.replace("</think><answer>", "</think>\n<answer>")

        lines = text.split("\n")
        start_y = 20
        for i, line in enumerate(lines):
            y = start_y + (i * 20)
            cv2.putText(
                white_box,
                line,
                (10, y),
                font,
                font_scale,
                font_color,
                thickness,
                line_type,
            )

        combined_frame = np.hstack((frame, white_box, plot_image_rgb))
        out.write(combined_frame)

    out.release()
