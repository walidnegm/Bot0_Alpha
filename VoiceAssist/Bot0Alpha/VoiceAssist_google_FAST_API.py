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
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException

voiceapp = FastAPI()

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
        self.running = False

    def start_capture(self):
        self.running = True
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
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
    def __init__(self, audio_capture, llm_client):
        self.audio_capture = audio_capture
        self.recognizer = sr.Recognizer()
        self.silence_start_time = None
        self.llm_client = llm_client
        self.tts_queue = Queue()
        self.stop_flag = False
        self.vector_embedding = vector_search()
        self.vector_embedding.load_embeddings()

    def is_silent(self, audio_segment):
        audio_data = np.frombuffer(audio_segment, np.int16)
        volume = np.max(audio_data).mean()
        return volume < SILENCE_THRESHOLD

    def transcribe_audio(self):
        audio_segments = []
        while self.audio_capture.running:
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
                            print(f"Wake-up word '{WAKE_UP_WORD}' detected.")
                            return text
                    except Exception as e:
                        print(f"Error during transcription: {e}")
                    finally:
                        audio_segments.clear()
                        self.silence_start_time = None

    def process_command(self, command):
        command_after_wake_up = command.split(WAKE_UP_WORD, 1)[1].strip()
        result = self.vector_embedding.search_vector_store(command_after_wake_up)
        response = self.llm_client.generate_completion(command_after_wake_up, result, mode="openai")
        if response:
            print("Assistant response:", response)
            self.tts_queue.put(response)
        return response
    

# WebSocket Endpoint for Real-Time Listening
@voiceapp.websocket("/listen/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    audio_capture = AudioCapture()
    llm_client = LLamaClientNV()
    transcriber = Transcriber(audio_capture, llm_client)

    # Start audio capture in a separate thread
    capture_thread = threading.Thread(target=audio_capture.start_capture, daemon=True)
    capture_thread.start()

    try:
        while True:
            transcription = transcriber.transcribe_audio()
            if transcription:
                response = transcriber.process_command(transcription)
                await websocket.send_json({"transcription": transcription, "response": response})
                if WAKE_UP_WORD in transcription.lower():
                    audio_capture.stop_capture()
                    break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        audio_capture.stop_capture()
        capture_thread.join()
        print("WebSocket connection closed.")

class TextInput(BaseModel):
    text: str

# Text-to-Speech Endpoint
@voiceapp.post("/text-to-speech/")
async def text_to_speech_task(background_tasks: BackgroundTasks, input: TextInput):
    """
    Handle TTS generation as a background task.
    """
    def generate_tts(text):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 130)
        tts_engine.setProperty("volume", 1.0)
        voices = tts_engine.getProperty('voices')
        tts_engine.setProperty('voice', voices[1].id)  # Female voice (index 1)
        tts_engine.say(text)
        tts_engine.runAndWait()
        print(f"TTS generated for: {text}")

    # Add the TTS generation to the background tasks
    background_tasks.add_task(generate_tts, input.text)

    return {"status": "TTS task started", "text": input.text}

# Health Check Endpoint
@voiceapp.get("/health-check/")
def health_check():
    return {"status": "Service running"}