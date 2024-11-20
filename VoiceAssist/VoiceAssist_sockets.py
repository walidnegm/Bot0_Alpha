import os
import time
import sys
import pyaudio
import pyttsx3
import openai
from llamav2 import LLamaClientNV
import numpy as np
import speech_recognition as sr
from queue import Queue
from search_vectors import vector_search  # Assuming this is your module
import configparser
from fastapi import FastAPI, UploadFile, BackgroundTasks, WebSocket, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import wave
import logging
from starlette.websockets import WebSocketDisconnect, WebSocketState

voiceapp = FastAPI()

# Constants
WAKE_UP_WORD = "jarvis"
DONE_WORD = "done"
SILENCE_THRESHOLD = 500
PAUSE_DURATION = 1.5
RATE = 16000  # Changed to 16000 for better speech recognition compatibility
CHUNK = 1024
CHANNELS = 1  # Changed to 1 for mono audio
SAMPLE_WIDTH = 2
FORMAT = pyaudio.paInt16
TIMEOUT_LIMIT = 10  # seconds
DEBUG_FILE_PATH = 'debug_audio.wav'
CAPTURE_DURATION = 8  # seconds to capture after wake-up word

logging.basicConfig(level=logging.DEBUG)

class Transcriber:
    def __init__(self, llm_client):
        self.recognizer = sr.Recognizer()
        self.silence_start_time = None
        self.llm_client = llm_client
        self.tts_queue = Queue()
        self.vector_embedding = vector_search()
        self.vector_embedding.load_embeddings()
        self.capture_active = False
        self.capture_start_time = None
        self.transcription_buffer = ""

    def is_silent(self, audio_segment):
        audio_data = np.frombuffer(audio_segment, np.int16)
        volume = np.max(audio_data)
        return volume < SILENCE_THRESHOLD

    def transcribe_audio_bytes(self, audio_data):
        """Transcribe raw audio bytes while detecting silence and wake-up word."""
        if not audio_data or len(audio_data) < CHUNK:
            logging.debug("Insufficient audio data received.")
            return None

        try:
            # Save audio data for debugging
            with wave.open(DEBUG_FILE_PATH, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(RATE)
                wav_file.writeframes(audio_data)
            
            # Silence detection
            if self.is_silent(audio_data):
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                elif time.time() - self.silence_start_time >= PAUSE_DURATION:
                    logging.debug("Silence detected for enough duration.")
            else:
                self.silence_start_time = None

            try:
                # Transcription
                audio = sr.AudioData(audio_data, RATE, SAMPLE_WIDTH)
                text = self.recognizer.recognize_google(audio)
                logging.debug(f"Recognized text: {text}")

                # Log all transcription
                logging.info(f"Transcription: {text}")

                # Wake-up word detection
                if WAKE_UP_WORD in text.lower():
                    logging.debug(f"Wake-up word '{WAKE_UP_WORD}' detected.")
                    self.capture_active = True
                    self.capture_start_time = time.time()
                    self.transcription_buffer = text
                    return None

                # Capture additional context after wake-up word
                if self.capture_active:
                    self.transcription_buffer += " " + text
                    if DONE_WORD in text.lower() or (time.time() - self.capture_start_time >= CAPTURE_DURATION):
                        self.capture_active = False
                        return self.transcription_buffer

                return None
            except sr.UnknownValueError:
                logging.debug("Speech not recognized")
                return None
            except sr.RequestError as e:
                logging.error(f"Speech recognition service error: {e}")
                return None

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return None

    def process_command(self, command):
        try:
            if WAKE_UP_WORD in command.lower():
                parts = command.lower().split(WAKE_UP_WORD, 1)
                if len(parts) > 1 and parts[1].strip():
                    command_after_wake_up = parts[1].strip()
                    result = self.vector_embedding.search_vector_store(command_after_wake_up)
                    response = self.llm_client.generate_completion(command_after_wake_up, result, mode="openai")
                    if response:
                        print("Assistant response:", response)
                        self.tts_queue.put(response)
                        self.voice_response(response)  # Voice out the response
                    return response
                else:
                    logging.error("No command detected after wake-up word.")
            else:
                logging.error("Wake-up word not found in the command.")
        except Exception as e:
            logging.error(f"Error processing command: {e}", exc_info=True)
        return None

    def voice_response(self, response):
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 130)
            tts_engine.setProperty("volume", 1.0)
            voices = tts_engine.getProperty('voices')
            tts_engine.setProperty('voice', voices[1].id)  # Female voice (index 1)
            tts_engine.say(response)
            tts_engine.runAndWait()
            #logging.debug(f"Voiced response: {response}")
        except Exception as e:
            logging.error(f"Error voicing response: {e}", exc_info=True)
            
@voiceapp.websocket("/listen/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.debug("WebSocket connection established.")
    
    llm_client = LLamaClientNV()
    transcriber = Transcriber(llm_client=llm_client)

    audio_buffer = bytearray()
    retry_count = 0
    
    # Define buffer sizes (in bytes)
   # Increase buffer sizes (in bytes)
    min_process_size = RATE * CHANNELS * SAMPLE_WIDTH * 8  # 4 seconds -> 8 seconds
    max_buffer_size = RATE * CHANNELS * SAMPLE_WIDTH * 12 # 5 seconds -> 10 seconds
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=TIMEOUT_LIMIT / 2
                )
                retry_count = 0

                if not data:
                    continue

                # Add new data to buffer
                audio_buffer.extend(data)
                current_size = len(audio_buffer)
                logging.debug(f"Buffer size: {current_size}/{max_buffer_size} bytes")

                # Process when we have enough data
                if current_size >= min_process_size:
                    try:
                        logging.debug(f"Processing audio buffer of size: {current_size}")
                        transcription = transcriber.transcribe_audio_bytes(bytes(audio_buffer))
                        
                        if transcription:
                            response = transcriber.process_command(transcription)
                            await websocket.send_json({
                                "transcription": transcription,
                                "response": response
                            })
                            # Keep last 1 second for overlap
                            overlap_size = int(RATE * CHANNELS * SAMPLE_WIDTH * 1)
                            audio_buffer = audio_buffer[-overlap_size:]
                            logging.debug(f"Successful transcription. Keeping {len(audio_buffer)} bytes for overlap")
                        else:
                            # If buffer is too large, keep last 3 seconds
                            if current_size > max_buffer_size:
                                keep_size = int(RATE * CHANNELS * SAMPLE_WIDTH * 12)  # Keep last 12 seconds
                                audio_buffer = audio_buffer[-keep_size:]
                                logging.debug(f"Buffer too large. Trimmed to {len(audio_buffer)} bytes")

                    except Exception as e:
                        logging.error(f"Error processing audio: {e}", exc_info=True)

            except asyncio.TimeoutError:
                logging.warning("No data received within the timeout period.")
                if retry_count >= 3:  # Example retry limit
                    break
                retry_count += 1
                await asyncio.sleep(1)  # Retry delay
                continue

            except WebSocketDisconnect:
                logging.debug("WebSocket disconnected.")
                break

    except Exception as e:
        logging.error(f"Unexpected error in websocket handler: {e}", exc_info=True)
        
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        # Process any remaining transcription after disconnection
        if transcriber.capture_active and transcriber.transcription_buffer:
            transcription = transcriber.transcription_buffer
            response = transcriber.process_command(transcription)
            logging.debug(f"Processed remaining transcription after disconnection: {transcription}")
            if response:
                logging.debug(f"Response: {response}")

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
        tts_engine.setProperty('voice', voices[1].id)  # Female
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

# Main entry to run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    try:
        print("Starting WebSocket server...")
        uvicorn.run(
            voiceapp,
            host="127.0.0.1",
            port=8000,
            log_level="debug",
            timeout_keep_alive=60,
            ws_ping_interval=20,
            ws_ping_timeout=30
        )
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)