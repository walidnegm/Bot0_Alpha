import threading
import pyaudio
import speech_recognition as sr
import numpy as np
import time
#from llama_cpp import Llama
import openai
import pyttsx3
from llamav2 import LLamaClientNV
import whisper
from queue import Queue
import os
from socket_helper import SocketHelper
from search_vectors import vector_search



# Wake-up word
WAKE_UP_WORD = "jarvis"
WHISPER_MODEL_PATH = "ggml-base.en.bin"


# Audio capture settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 350
PAUSE_DURATION = 1.5

payload = {
    "source": "orin_ll",
    "destination": "carla",
    "data": "",
    "application": "steering"
}

class AudioCapture:
    def __init__(self):
        self.audio_data = []
        self.lock = threading.Lock()
        self.running = True

    def start_capture(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, 
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        print("Audio capture started.")
        while self.running:
            data = stream.read(CHUNK)
            with self.lock:
                self.audio_data.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()


    def stop_capture(self):
        self.running = False



#class LlamaClient:
#    def __init__(self, model):
#        self.llm = Llama(
#            model_path="llama-3-2-8b-Instruct-Q8_0.gguf",
#           chat_format="llama-3"
#        )
#        self.messages = [
#           {"role": "system", "content": "You are an assistant who perfectly describes images."}
#        ]

    # Function to interact with the model
#    def ask_llama(self, prompt):
#        self.messages.append({"role": "user", "content": prompt})
#       response = self.llm.create_chat_completion(self.messages)
#        self.messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
#        return response['choices'][0]['message']['content']


class Transcriber:
    def __init__(self, audio_capture, llm_client):
        self.audio_capture = audio_capture
        self.recognizer = sr.Recognizer()
        self.silence_start_time = None
        self.audio_segments = []
        self.tts_queue = []  # List to manage TTS queue
        self.tts_lock = threading.Lock()  # Lock for TTS queue access
        self.vector_embedding = vector_search
        self.llm = llm_client

    def is_silent(self, media_segment):
        audio_data = np.frombuffer(audio_segment, dtype=np.int16)
        volume = np.max(audio_data).mean()
        #print("Volume:", volume)
        return volume < SILENCE_THRESHOLD

    def transcribe_audio_whisper(self):
        print("Whisper Transcription started.")
        audio_segments = []  # Initialize outside the loop to accumulate audio
        while True:
            time.sleep(0.1)  # Short sleep to avoid busy waiting

            with self.audio_capture.lock:
                if self.audio_capture.audio_data:
                    audio_segments.extend(self.audio_capture.audio_data)
                    self.audio_capture.audio_data.clear()  # Clear the audio data after copying

            # Check if the last segment is silent
            if audio_segmnets: 
                last_segment = audio_segments[-1]
                if self.is_silent(last_segment):
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                else:
                    self.silence_start_time = None

                # Check if silence duration is sufficient
                if self.silence_start_time is not None and (time.time() - self.silence_start_time) >= PAUSE_DURATION:
                    combined_audio = b''.join(audio_segments)  # Combine audio segments
                    print("Testing...")

                    #audio_data = sr.AudioData(combined_audio, RATE, 2)

                    try:
                        audio_np = np.frombuffer(combined_audio, dtype=np.int16)
                        audio_np = audio_np.astype(np.float32) / 32767.0
                        print(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")

                        result= self.base_model.transcribe(audio_np, language='en')
                        text=result["text"]
            
                        print("Recognized text:", {text})
                        if WAKE_UP_WORD in text.lower():
                            self.process_command(text)

                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; {0}".format(e))

                    # Clear the buffer after processing

                    audio_segments.clear()
                    self.silence_start_time = None  # Reset silence timer

    def process_command(self, command):
        command = command.lower()
        if WAKE_UP_WORD in command:
            command_after_wake_up = command.split(WAKE_UP_WORD, 1)[1].strip()
            if command_after_wake_up:
                print (f"Proessing command: {command_after_wake_up}")
                with self.tts_lock:
                    self.tts_queue.append("okay")
                results = self.vector_embedding.vector_search(command_after_wake_up)

                if response:
                    print("Assistant response:", response)
                    with self.tts_lock:
                        self.tts_queue.append(response)
                else:
                    print("no command found after the wakeup word")
            
    def _tts_worker(self):

        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rae", 150)  # Speed (words per minute)
        tts_engine.setProperty("volume", 1.0)  # Volume (0 to 1)

        with self.tts_lock:
            if self.tts_queue:
                print ("QUEUE:", self.tts_queue)
                text = self.tts_queue.pop(0)  # Get the first item in the queue
                payload["data"] = text
                tts_engine.say(text)
                tts_engine.runAndWait()  # This will block until speech is finished

def main():

    audio_capture = AudioCapture()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    chat_client = LLamaClientNV(base_url="http://localhost:8080")
    transcriber = Transcriber(audio_capture, chat_client)


    # Start audio capture in a separate thread

    capture_thread = threading.Thread(target=audio_capture.start_capture)
    capture_thread.start()



    # Start transcription in a separate thread

    transcribe_thread = threading.Thread(target=transcriber.transcribe_audio)
    transcribe_thread.start()

    tts_thread = threading.Thread(target=transcriber._tts_worker)
    tts_thread.start()



    try:
        while True:
            time.sleep(3)  # Keep the main thread alive

    except KeyboardInterrupt:
        audio_capture.stop_capture()
        capture_thread.join()
        transcribe_thread.join()
        tts_thread.join()

if __name__ == "__main__":
    main()  