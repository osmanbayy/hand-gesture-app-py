import threading
import time
from collections import deque
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from .config import (
    LEFT_PANEL_WIDTH,
    RIGHT_PANEL_WIDTH,
    VIDEO_SIZE,
    WINDOW_MAXIMIZED_ON_START,
    WINDOW_MIN_SIZE,
    WINDOW_SIZE,
)
from .recognizer import HandGestureRecognizer
from .text_constants import (
    BUTTON_START,
    BUTTON_STOP,
    CAMERA_OPEN_ERROR,
    APP_HEADER_SUBTITLE,
    APP_HEADER_TITLE,
    HISTORY_TITLE,
    INFO_TEXT,
    LABEL_CONFIDENCE_PREFIX,
    LABEL_FPS_PREFIX,
    LABEL_GESTURE_PREFIX,
    LABEL_GESTURE_WAITING,
    LABEL_NO_HAND,
    PANEL_TITLE,
    SHORTCUT_HINT,
    STATUS_CAMERA_ERROR,
    STATUS_READY,
    STATUS_RUNNING,
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
        if WINDOW_MAXIMIZED_ON_START:
            self.after(10, lambda: self.state("zoomed"))

        self.running = False
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.recognizer = HandGestureRecognizer()

        self.latest_image: Optional[ImageTk.PhotoImage] = None
        self.current_gesture = LABEL_GESTURE_WAITING
        self.current_confidence = 0.0
        self.fps = 0.0
        self.gesture_history = deque(maxlen=10)
        self._last_history_gesture = ""

        self._build_layout()
        self._bind_shortcuts()
        self._set_status(STATUS_READY, "#1f6aa5")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        # Left panel grows/shrinks with window size, right panel stays stable.
        self.grid_columnconfigure(0, weight=1, minsize=LEFT_PANEL_WIDTH)
        self.grid_columnconfigure(1, weight=0, minsize=RIGHT_PANEL_WIDTH)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self, corner_radius=12, fg_color="#111827")
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(14, 0))
        self.header.grid_columnconfigure(0, weight=1)

        header_title = ctk.CTkLabel(
            self.header,
            text=APP_HEADER_TITLE,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header_title.grid(row=0, column=0, padx=16, pady=(10, 0), sticky="w")

        header_subtitle = ctk.CTkLabel(
            self.header,
            text=APP_HEADER_SUBTITLE,
            text_color="#9ca3af",
            font=ctk.CTkFont(size=13),
        )
        header_subtitle.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="w")

        self.status_badge = ctk.CTkLabel(
            self.header,
            text=STATUS_READY,
            corner_radius=14,
            fg_color="#1f6aa5",
            text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=12,
            pady=4,
        )
        self.status_badge.grid(row=0, column=1, rowspan=2, padx=16, pady=10, sticky="e")

        self.video_frame = ctk.CTkFrame(self, corner_radius=14)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self.video_frame.grid_propagate(False)
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
        side_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 16), pady=16)
        side_panel.grid_propagate(False)
        side_panel.grid_columnconfigure(0, weight=1)
        side_panel.grid_rowconfigure(8, weight=1)

        panel_title = ctk.CTkLabel(
            side_panel,
            text=PANEL_TITLE,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        panel_title.grid(row=0, column=0, padx=16, pady=(18, 8), sticky="w")

        self.gesture_card = ctk.CTkFrame(side_panel, corner_radius=10, fg_color="#1f2937")
        self.gesture_card.grid(row=1, column=0, padx=16, pady=(6, 8), sticky="ew")
        self.gesture_card.grid_columnconfigure(0, weight=1)
        self.gesture_label = ctk.CTkLabel(
            self.gesture_card,
            text=f"{LABEL_GESTURE_PREFIX}: {LABEL_GESTURE_WAITING}",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.gesture_label.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self.confidence_card = ctk.CTkFrame(side_panel, corner_radius=10, fg_color="#1f2937")
        self.confidence_card.grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        self.confidence_card.grid_columnconfigure(0, weight=1)
        self.confidence_label = ctk.CTkLabel(
            self.confidence_card,
            text=f"{LABEL_CONFIDENCE_PREFIX}: %0",
            font=ctk.CTkFont(size=16),
        )
        self.confidence_label.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self.fps_card = ctk.CTkFrame(side_panel, corner_radius=10, fg_color="#1f2937")
        self.fps_card.grid(row=3, column=0, padx=16, pady=8, sticky="ew")
        self.fps_card.grid_columnconfigure(0, weight=1)
        self.fps_label = ctk.CTkLabel(
            self.fps_card,
            text=f"{LABEL_FPS_PREFIX}: 0.0",
            font=ctk.CTkFont(size=16),
        )
        self.fps_label.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self.start_button = ctk.CTkButton(
            side_panel,
            text=BUTTON_START,
            height=44,
            command=self.start_camera,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.start_button.grid(row=4, column=0, padx=16, pady=(16, 8), sticky="ew")

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

        history_title = ctk.CTkLabel(
            side_panel,
            text=HISTORY_TITLE,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        history_title.grid(row=6, column=0, padx=16, pady=(12, 4), sticky="w")

        self.history_box = ctk.CTkTextbox(side_panel, height=130, corner_radius=10, wrap="word")
        self.history_box.grid(row=7, column=0, padx=16, pady=(0, 8), sticky="nsew")
        self.history_box.insert("1.0", "-")
        self.history_box.configure(state="disabled")

        info_label = ctk.CTkLabel(
            side_panel,
            text=INFO_TEXT,
            justify="left",
            font=ctk.CTkFont(size=13),
        )
        info_label.grid(row=8, column=0, padx=16, pady=(4, 10), sticky="nw")

        shortcut_hint = ctk.CTkLabel(
            side_panel,
            text=SHORTCUT_HINT,
            text_color="#9ca3af",
            font=ctk.CTkFont(size=12),
        )
        shortcut_hint.grid(row=9, column=0, padx=16, pady=(0, 16), sticky="w")

    def _bind_shortcuts(self) -> None:
        self.bind("<space>", self._on_space_key)
        self.bind("<Escape>", self._on_escape_key)

    def _on_space_key(self, _event=None) -> None:
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def _on_escape_key(self, _event=None) -> None:
        self._on_close()

    def _set_status(self, text: str, color: str) -> None:
        self.status_badge.configure(text=text, fg_color=color)

    @staticmethod
    def _cv_safe_text(text: str) -> str:
        # cv2.putText does not render Turkish characters reliably with default Hershey fonts.
        replacements = str.maketrans({
            "ç": "c", "Ç": "C",
            "ğ": "g", "Ğ": "G",
            "ı": "i", "İ": "I",
            "ö": "o", "Ö": "O",
            "ş": "s", "Ş": "S",
            "ü": "u", "Ü": "U",
        })
        return text.translate(replacements)

    def _push_history(self, gesture: str, confidence: float) -> None:
        if gesture == self._last_history_gesture:
            return

        self._last_history_gesture = gesture
        timestamp = time.strftime("%H:%M:%S")
        self.gesture_history.appendleft(f"[{timestamp}] {gesture} (%{int(confidence * 100)})")

        self.history_box.configure(state="normal")
        self.history_box.delete("1.0", "end")
        if self.gesture_history:
            self.history_box.insert("1.0", "\n".join(self.gesture_history))
        else:
            self.history_box.insert("1.0", "-")
        self.history_box.configure(state="disabled")

    def start_camera(self) -> None:
        if self.running:
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.video_label.configure(text=CAMERA_OPEN_ERROR)
            self._set_status(STATUS_CAMERA_ERROR, "#b91c1c")
            return

        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_status(STATUS_RUNNING, "#15803d")

        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

    def stop_camera(self) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._set_status(STATUS_READY, "#1f6aa5")
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
                self._cv_safe_text(f"{LABEL_GESTURE_PREFIX}: {self.current_gesture}"),
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                self._cv_safe_text(f"{LABEL_CONFIDENCE_PREFIX}: %{int(self.current_confidence * 100)}"),
                (20, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            target_w = self.video_label.winfo_width()
            target_h = self.video_label.winfo_height()
            if target_w <= 1 or target_h <= 1:
                target_w, target_h = VIDEO_SIZE
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            self.latest_image = ImageTk.PhotoImage(image=image)

            self.after(0, self._update_ui)

        self.after(0, self._set_idle_screen)

    def _update_ui(self) -> None:
        if self.latest_image is not None:
            self.video_label.configure(image=self.latest_image, text="")
        self.gesture_label.configure(text=f"{LABEL_GESTURE_PREFIX}: {self.current_gesture}")
        self.confidence_label.configure(text=f"{LABEL_CONFIDENCE_PREFIX}: %{int(self.current_confidence * 100)}")
        self.fps_label.configure(text=f"{LABEL_FPS_PREFIX}: {self.fps:.1f}")
        self._push_history(self.current_gesture, self.current_confidence)

    def _set_idle_screen(self) -> None:
        self.video_label.configure(image=None, text=VIDEO_STOPPED)
        self.gesture_label.configure(text=f"{LABEL_GESTURE_PREFIX}: {LABEL_GESTURE_WAITING}")
        self.confidence_label.configure(text=f"{LABEL_CONFIDENCE_PREFIX}: %0")
        self.fps_label.configure(text=f"{LABEL_FPS_PREFIX}: 0.0")

    def _on_close(self) -> None:
        self.stop_camera()
        self.recognizer.close()
        self.destroy()
