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
import wave

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
    def __init__(self, llm_client):
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

    def transcribe_audio_bytes(self, audio_data):
        """
        Transcribe raw audio bytes while detecting silence and wake-up word.
        """
        if not audio_data or len(audio_data) < CHUNK:
            print("Insufficient audio data received.")
            # Still proceed to save the data for debugging

        # Save valid audio data to a debug file for testing
        try:
            with wave.open("debug_audio.wav", "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(RATE)  # Sampling rate
                wav_file.writeframes(audio_data)
            print("Audio data saved to 'debug_audio.wav' for debugging.")
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return None

        try:
            # Silence detection
            if self.is_silent(audio_data):
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                elif time.time() - self.silence_start_time >= PAUSE_DURATION:
                    print("Silence detected for enough duration. Ready to process.")
            else:
                self.silence_start_time = None

        # Transcription
            audio = sr.AudioData(audio_data, RATE, 2)
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")

        # Wake-up word detection
            if WAKE_UP_WORD in text.lower():
                print(f"Wake-up word '{WAKE_UP_WORD}' detected.")
                return text  # Can trigger additional processing
            return None
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

def process_command(self, command):
    command_after_wake_up = command.split(WAKE_UP_WORD, 1)[1].strip()
    result = self.vector_embedding.search_vector_store(command_after_wake_up)
    response = self.llm_client.generate_completion(command_after_wake_up, result, mode="openai")
    if response:
        print("Assistant response:", response)
        self.tts_queue.put(response)
    return response
    

@voiceapp.websocket("/listen/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    llm_client = LLamaClientNV()
    transcriber = Transcriber(llm_client=llm_client)

    audio_buffer = bytearray()  # To collect chunks for processing
    try:
        while True:
            # Receive audio chunks from the client
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            # Process audio once the buffer reaches a reasonable size
            if len(audio_buffer) > RATE * 2:  # Example: 1 second of audio
                transcription = transcriber.transcribe_audio_bytes(audio_buffer)
                if transcription:
                    response = transcriber.process_command(transcription)
                    await websocket.send_json({"transcription": transcription, "response": response})
                    if WAKE_UP_WORD in transcription.lower():
                        break  # Stop processing on wake-up word
                audio_buffer.clear()  # Clear buffer after processing
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
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