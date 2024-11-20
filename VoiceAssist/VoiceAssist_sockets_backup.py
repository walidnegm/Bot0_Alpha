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
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
DEBUG_FILE_PATH = 'debug_audio.wav'
DEBUG_FILE_PATH_1 = 'debug_audio_1.wav'
DEBUG_FILE_PATH_2 = 'debug_audio_2.wav'
BACKUP_FILE_PATH = 'debug_audio_backup.wav'
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
 
def check_and_rotate_audio_files():
    """Check and rotate between two audio files based on their size."""
    if os.path.exists(DEBUG_FILE_PATH_1) and os.path.getsize(DEBUG_FILE_PATH_1) > MAX_FILE_SIZE:
        # Rotate the first file to backup if it exceeds the size limit
        print(f"{DEBUG_FILE_PATH_1} is full. Moving to backup.")
        os.rename(DEBUG_FILE_PATH_1, BACKUP_FILE_PATH)  # Move the old file to backup
        # Create a new file to start recording
        open(DEBUG_FILE_PATH_1, 'w').close()
        
    elif os.path.exists(DEBUG_FILE_PATH_2) and os.path.getsize(DEBUG_FILE_PATH_2) > MAX_FILE_SIZE:
        # If the second file is full, overwrite the first file
        print(f"{DEBUG_FILE_PATH_2} is full. Moving to {DEBUG_FILE_PATH_1}.")
        os.rename(DEBUG_FILE_PATH_2, DEBUG_FILE_PATH_1)  # Move file 2 to file 1
        # Create a new second file to start recording
        open(DEBUG_FILE_PATH_2, 'w').close()
    
    # If both files are too large, delete the old backup
    if os.path.exists(BACKUP_FILE_PATH) and os.path.getsize(BACKUP_FILE_PATH) > MAX_FILE_SIZE:
        print("Both files are full, deleting backup.")
        os.remove(BACKUP_FILE_PATH)
        
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
            return None
        # Save valid audio data to a debug file for testing
        try:
            check_and_rotate_audio_files()  # Check and rotate if needed
            with wave.open(DEBUG_FILE_PATH_1, "wb") as wav_file:
                wav_file.setnchannels(2)  # Mono audio
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