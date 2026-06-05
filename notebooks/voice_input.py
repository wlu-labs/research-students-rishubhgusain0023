#!/usr/bin/env python3
# coding=utf-8
"""
voice_input.py
--------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

Voice-to-task pipeline:
  1. Listens for voice input via microphone
  2. Uses OpenAI Whisper to transcribe speech to text
  3. Publishes the transcribed text to /task topic
  4. NavGPT + RRT* pipeline takes over from there

Run:
  python3 /home/jetson/Desktop/Rishubh/voice_input.py

Then speak a command like:
  "find the chair"
  "find a bottle"
  "go to the person"
"""

import pyaudio
import whisper
import wave
import os
import tempfile
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ─────────────── CONFIG ───────────────────────────────────────
WHISPER_MODEL   = "base"      # tiny / base / small (base is fast on Jetson)
RECORD_SECONDS  = 4           # how long to record each command
SAMPLE_RATE     = 16000       # Hz — Whisper expects 16kHz
CHUNK           = 1024        # audio buffer size
CHANNELS        = 1           # mono
SILENCE_THRESH  = 500         # amplitude threshold to detect speech
# ──────────────────────────────────────────────────────────────


class VoiceInputNode(Node):

    def __init__(self):
        super().__init__("voice_input")

        self.task_pub = self.create_publisher(String, '/task', 10)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  VoiceInputNode ready")
        self.get_logger().info(f"  Whisper model : {WHISPER_MODEL}")
        self.get_logger().info(f"  Record time   : {RECORD_SECONDS}s per command")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Loading Whisper model...")

        self.model = whisper.load_model(WHISPER_MODEL)
        self.get_logger().info("Whisper ready. Listening for voice commands...")
        self.get_logger().info("Speak a command after the prompt appears.")

        # Start listening loop
        self.create_timer(0.1, self._check_ready)
        self._ready = True

    def _check_ready(self):
        if self._ready:
            self._ready = False
            self._listen_and_publish()
            self._ready = True

    def _listen_and_publish(self):
        print("\n[VOICE] Listening... speak now")

        # Record audio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        frames = []
        for _ in range(int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save to temp wav file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wf = wave.open(tmp.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Transcribe with Whisper
        print("[VOICE] Transcribing...")
        result = self.model.transcribe(tmp.name, language="en", fp16=False)
        text   = result["text"].strip().lower()
        os.unlink(tmp.name)

        if not text or len(text) < 2:
            print("[VOICE] Nothing detected, listening again...")
            return

        print(f"[VOICE] Heard: '{text}'")
        self.get_logger().info(f"[VOICE] Transcribed: '{text}'")

        # Publish to /task
        msg = String()
        msg.data = text
        self.task_pub.publish(msg)
        self.get_logger().info(f"[VOICE] Published to /task: '{text}'")
        print(f"[VOICE] → /task: '{text}'")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()