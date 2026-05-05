import threading
import time
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from .config import VIDEO_SIZE, WINDOW_MIN_SIZE, WINDOW_SIZE
from .recognizer import HandGestureRecognizer
from .text_constants import (
    BUTTON_START,
    BUTTON_STOP,
    CAMERA_OPEN_ERROR,
    INFO_TEXT,
    LABEL_CONFIDENCE_PREFIX,
    LABEL_FPS_PREFIX,
    LABEL_GESTURE_PREFIX,
    LABEL_GESTURE_WAITING,
    LABEL_NO_HAND,
    PANEL_TITLE,
    VIDEO_IDLE_HINT,
    VIDEO_STOPPED,
    VIDEO_TITLE,
    WINDOW_TITLE,
)


class GestureApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(*WINDOW_MIN_SIZE)

        self.running = False
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.recognizer = HandGestureRecognizer()

        self.latest_image: Optional[ImageTk.PhotoImage] = None
        self.current_gesture = LABEL_GESTURE_WAITING
        self.current_confidence = 0.0
        self.fps = 0.0

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.video_frame = ctk.CTkFrame(self, corner_radius=14)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.video_frame.grid_rowconfigure(1, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.video_frame,
            text=VIDEO_TITLE,
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="w")

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text=VIDEO_IDLE_HINT,
            font=ctk.CTkFont(size=16),
            corner_radius=12,
        )
        self.video_label.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")

        side_panel = ctk.CTkFrame(self, corner_radius=14)
        side_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        side_panel.grid_columnconfigure(0, weight=1)

        panel_title = ctk.CTkLabel(
            side_panel,
            text=PANEL_TITLE,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        panel_title.grid(row=0, column=0, padx=16, pady=(18, 8), sticky="w")

        self.gesture_label = ctk.CTkLabel(
            side_panel,
            text=f"{LABEL_GESTURE_PREFIX}: {LABEL_GESTURE_WAITING}",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.gesture_label.grid(row=1, column=0, padx=16, pady=8, sticky="w")

        self.confidence_label = ctk.CTkLabel(
            side_panel,
            text=f"{LABEL_CONFIDENCE_PREFIX}: %0",
            font=ctk.CTkFont(size=16),
        )
        self.confidence_label.grid(row=2, column=0, padx=16, pady=8, sticky="w")

        self.fps_label = ctk.CTkLabel(
            side_panel,
            text=f"{LABEL_FPS_PREFIX}: 0.0",
            font=ctk.CTkFont(size=16),
        )
        self.fps_label.grid(row=3, column=0, padx=16, pady=8, sticky="w")

        self.start_button = ctk.CTkButton(
            side_panel,
            text=BUTTON_START,
            height=44,
            command=self.start_camera,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.start_button.grid(row=4, column=0, padx=16, pady=(20, 8), sticky="ew")

        self.stop_button = ctk.CTkButton(
            side_panel,
            text=BUTTON_STOP,
            height=44,
            command=self.stop_camera,
            state="disabled",
            fg_color="#b71c1c",
            hover_color="#8e0000",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.stop_button.grid(row=5, column=0, padx=16, pady=8, sticky="ew")

        info_label = ctk.CTkLabel(
            side_panel,
            text=INFO_TEXT,
            justify="left",
            font=ctk.CTkFont(size=14),
        )
        info_label.grid(row=6, column=0, padx=16, pady=(18, 16), sticky="nw")

    def start_camera(self) -> None:
        if self.running:
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.video_label.configure(text=CAMERA_OPEN_ERROR)
            return

        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

    def stop_camera(self) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _camera_loop(self) -> None:
        last_time = time.time()

        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            hand_landmarks_list = self.recognizer.process(frame)

            if hand_landmarks_list:
                hand_landmarks = hand_landmarks_list[0]
                self.recognizer.draw_hand(frame, hand_landmarks)
                gesture = self.recognizer.classify(hand_landmarks)
                self.current_gesture = gesture.label
                self.current_confidence = gesture.confidence
            else:
                self.current_gesture = LABEL_NO_HAND
                self.current_confidence = 0.0

            current_time = time.time()
            delta = max(current_time - last_time, 1e-6)
            self.fps = 1.0 / delta
            last_time = current_time

            cv2.putText(
                frame,
                f"{LABEL_GESTURE_PREFIX}: {self.current_gesture}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{LABEL_CONFIDENCE_PREFIX}: %{int(self.current_confidence * 100)}",
                (20, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize(VIDEO_SIZE, Image.Resampling.LANCZOS)
            self.latest_image = ImageTk.PhotoImage(image=image)

            self.after(0, self._update_ui)

        self.after(0, self._set_idle_screen)

    def _update_ui(self) -> None:
        if self.latest_image is not None:
            self.video_label.configure(image=self.latest_image, text="")
        self.gesture_label.configure(text=f"{LABEL_GESTURE_PREFIX}: {self.current_gesture}")
        self.confidence_label.configure(text=f"{LABEL_CONFIDENCE_PREFIX}: %{int(self.current_confidence * 100)}")
        self.fps_label.configure(text=f"{LABEL_FPS_PREFIX}: {self.fps:.1f}")

    def _set_idle_screen(self) -> None:
        self.video_label.configure(image=None, text=VIDEO_STOPPED)
        self.gesture_label.configure(text=f"{LABEL_GESTURE_PREFIX}: {LABEL_GESTURE_WAITING}")
        self.confidence_label.configure(text=f"{LABEL_CONFIDENCE_PREFIX}: %0")
        self.fps_label.configure(text=f"{LABEL_FPS_PREFIX}: 0.0")

    def _on_close(self) -> None:
        self.stop_camera()
        self.recognizer.close()
        self.destroy()
