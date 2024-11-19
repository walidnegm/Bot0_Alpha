import os
import threading
import time
import sys
import pyaudio
import pyttsx3
import openai
from llamav2 import LLamaClientNV
import numpy as np
import speech_recognition as sr
import whisper
from queue import Queue
from socket_helper import SocketHelper
from search_vectors import vector_search  # Assuming this is your module
import openai
import configparser
from gtts import gTTS


# Constants
WAKE_UP_WORD = "jarvis"
SILENCE_THRESHOLD = 500
PAUSE_DURATION = 1.5
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16

WHISPER_MODEL_PATH="ggml_base.en.bin"

payload = {
    "source": "orin-11m",
    "destination": "carla",
    "data":"",
    "application": "steering"}
 
#from globals import setup_global_openai_client

class AudioCapture:
    def __init__(self):
        self.audio_data = []
        self.lock = threading.Lock()
        self.running = True

    def start_capture(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Audio capture started.")

        try:
            while self.running:
                data = stream.read(CHUNK)
                with self.lock:
                    self.audio_data.append(data)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            print("Audio capture stopped.")

    def stop_capture(self):
        self.running = False


class Transcriber:
    def __init__(self, audio_capture, llmClient):
        self.audio_capture = audio_capture
        self.recognizer = sr.Recognizer()
        self.silence_start_time = None
        self.llmClient = llmClient
        self.base_model = whisper.load_model("base")
        self.tts_queue = []
        self.tts_lock = threading.Lock()
        self.vector_embedding = vector_search()
        self.vector_embedding.load_embeddings()
        self.stop_flag = False

    def is_silent(self, audio_segment):
        audio_data = np.frombuffer(audio_segment, np.int16)
        volume = np.max(audio_data).mean()
        return volume < SILENCE_THRESHOLD

    def transcribe_audio_whisper(self):
        print("Whisper Transcription started.")
        audio_segments = []
        while not self.stop_flag:
            time.sleep(0.1)

            with self.audio_capture.lock:
                if self.audio_capture.audio_data:
                    audio_segments.extend(self.audio_capture.audio_data)
                    self.audio_capture.audio_data.clear()

            if audio_segments:
                last_segment = audio_segments[-1]
                if self.is_silent(last_segment):
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                else:
                    self.silence_start_time = None

                if self.silence_start_time is not None and (time.time() - self.silence_start_time) >= PAUSE_DURATION:
                    combined_audio = b''.join(audio_segments)
                    print("Processing audio...")
                    try:
                        audio_np = np.frombuffer(combined_audio, dtype=np.int16)
                        audio_np = audio_np.astype(np.float32) / 32768.0
                        print(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")

                        result = self.base_model.transcribe(audio_np, language='en')
                        text = result["text"]
                        print("Recognized text:", text)

                        if WAKE_UP_WORD in text.lower():
                            self.process_command(text)
                            
                    except Exception as e:
                        print(f"Error during transcription: {e}")
                    finally:
                        audio_segments.clear()
                        self.silence_start_time = None

        print("Stopping Whisper transcription...")

    def transcribe_audio(self):
        print("Transcription started.")
        audio_segments = []
        while not self.stop_flag:
            time.sleep(0.1)

            with self.audio_capture.lock:
                if self.audio_capture.audio_data:
                    audio_segments.extend(self.audio_capture.audio_data)
                    self.audio_capture.audio_data.clear()

            if audio_segments:
                last_segment = audio_segments[-1]
                if self.is_silent(last_segment):
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                else:
                    self.silence_start_time = None

                if self.silence_start_time is not None and (time.time() - self.silence_start_time) >= PAUSE_DURATION:
                    combined_audio = b''.join(audio_segments)
                    try:
                        audio_data = sr.AudioData(combined_audio, RATE, 2)
                        text = self.recognizer.recognize_google(audio_data)
                        print(f"Recognized text: {text}")

                        if WAKE_UP_WORD in text.lower():
                            self.process_command(text)
                    except Exception as e:
                        print(f"Error during Google transcription: {e}")
                    finally:
                        audio_segments.clear()
                        self.silence_start_time = None

        print("Stopping transcription...")

    def process_command(self, command):
        command = command.lower()
        if WAKE_UP_WORD in command:
            command_after_wake_up = command.split(WAKE_UP_WORD, 1)[1].strip()
            if command_after_wake_up:
                print(f"Processing command: {command_after_wake_up}")
                with self.tts_lock:
                    self.tts_queue.append("okay")
                result = self.vector_embedding.search_vector_store(command_after_wake_up)
                response = self.llmClient.generate_completion(command_after_wake_up, result, mode="openai")

                if response:
                    print("Assistant response:", response)
                    with self.tts_lock:
                        self.tts_queue.append(response)

    def _tts_worker(self):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 130)
        tts_engine.setProperty("volume", 1.0)
        voices = tts_engine.getProperty('voices')
        tts_engine.setProperty('voice', voices[1].id)  # Female voice (usually index 1)

        while not self.stop_flag:
            with self.tts_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
                else:
                    text = None
            if text:
                tts_engine.say(text)
                tts_engine.runAndWait()
            else:
                time.sleep(0.1)

        print("Stopping TTS worker...")

def main():
    audio_capture = AudioCapture()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    #setup_global_openai_client(config_path="./config.ini")

    # Initialize the client
    chat_client = LLamaClientNV()

    transcriber = Transcriber(audio_capture, chat_client)

    capture_thread = threading.Thread(target=audio_capture.start_capture, daemon=True)
    transcribe_thread = threading.Thread(target=transcriber.transcribe_audio_whisper, daemon=True)
    tts_thread = threading.Thread(target=transcriber._tts_worker, daemon=True)

    capture_thread.start()
    transcribe_thread.start()
    tts_thread.start()

    try:
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nCtrl-C received. Shutting down...")
        audio_capture.stop_capture()
        transcriber.stop_flag = True

        capture_thread.join(timeout=5)
        transcribe_thread.join(timeout=5)
        tts_thread.join(timeout=5)

        print("All threads stopped. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
