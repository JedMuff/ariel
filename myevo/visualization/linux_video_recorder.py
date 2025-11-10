"""Linux-compatible video recorder.

This is a wrapper around ARIEL's VideoRecorder that uses XVID or MJPEG codec
which are more commonly available on Linux systems.
"""

import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy import typing as npt


class LinuxVideoRecorder:
    """Video recorder optimized for Linux compatibility."""

    # Use XVID codec which is widely available on Linux
    # Fall back to MJPEG if XVID is not available
    _video_encoding: str = "XVID"
    _add_timestamp_to_file_name: bool = True
    _file_extension: str = ".avi"  # AVI is more compatible with XVID

    def __init__(
        self,
        file_name: str = "video",
        output_folder: str | Path | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Create a video recording compatible with Linux.

        Parameters
        ----------
        file_name
            Name of the video file.
        output_folder
            Folder where the video will be saved.
            If None, saves to current directory.
        width
            Width of the video frames.
        height
            Height of the video frames.
        fps
            Frames per second for the video.
        """
        # Save local variables
        self.width = width
        self.height = height
        self.fps = fps

        # Set output folder
        if output_folder is None:
            output_folder = Path.cwd()
        elif isinstance(output_folder, str):
            output_folder = Path(output_folder)

        # Ensure output folder exists
        Path(output_folder).mkdir(exist_ok=True, parents=True)

        # Generate video name
        if self._add_timestamp_to_file_name:
            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y%m%d_%H%M%S",
            )
            file_name += f"_{timestamp}"
        output_file = output_folder / f"{file_name}{self._file_extension}"

        # Store output file path
        self.output_file = str(output_file)

        # Try XVID first, fall back to MJPEG if not available
        video_writer = None

        # Try XVID
        fourcc = cv2.VideoWriter_fourcc(*self._video_encoding)
        video_writer = cv2.VideoWriter(
            self.output_file,
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        # Check if VideoWriter was successfully initialized
        if not video_writer.isOpened():
            print("XVID codec not available, falling back to MJPEG")
            # Fall back to MJPEG
            self._video_encoding = "MJPG"
            fourcc = cv2.VideoWriter_fourcc(*self._video_encoding)
            video_writer = cv2.VideoWriter(
                self.output_file,
                fourcc,
                self.fps,
                (self.width, self.height),
            )

            # If MJPEG also fails, raise error
            if not video_writer.isOpened():
                raise RuntimeError(
                    f"Failed to initialize video writer with both XVID and MJPEG codecs. "
                    f"Output file: {self.output_file}, Resolution: {self.width}x{self.height}, FPS: {self.fps}"
                )

        # Class attributes
        self.frame_count = 0
        self.video_writer = video_writer

    def write(self, frame: npt.ArrayLike) -> None:
        """Write MuJoCo frame to video.

        Parameters
        ----------
        frame
            Frame to write to the video.
        """
        # Convert PIL Image to numpy array (OpenCV uses BGR format)
        opencv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Save frame
        self.video_writer.write(opencv_image)

        # Increment frame counter
        self.frame_count += 1

    def release(self) -> None:
        """Close video writer and save video locally."""
        self.video_writer.release()
