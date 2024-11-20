import asyncio
import websockets
import wave
from pydub import AudioSegment

async def test_websocket():
    uri = "ws://127.0.0.1:8000/listen/"
    
    # Original audio file
    audio_file = "NivineDrivingSupport.wav"
    converted_file = "VoicewithJarvis_converted.wav"
    
    # Convert the audio to 16kHz, Mono if needed
    with wave.open(audio_file, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        if frame_rate != 16000 or channels != 1:
            print("Converting audio to 16kHz, mono...")
            sound = AudioSegment.from_wav(audio_file)
            sound = sound.set_channels(1)
            sound = sound.set_frame_rate(16000)
            sound.export(converted_file, format="wav")
            audio_file = converted_file  # Update to the converted file

    # Open the (possibly converted) audio file
    with wave.open(audio_file, "rb") as wav_file:
        async with websockets.connect(uri) as websocket:
            chunk_size = 4096
            audio_data = wav_file.readframes(wav_file.getnframes())
            
            # Split the full audio data into 4096-byte chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) == 0:
                    break

                # Ensure chunk is exactly 4096 bytes before sending
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))

                try:
                    print(f"Sending {len(chunk)} bytes of audio...")
                    await websocket.send(chunk)
                    await asyncio.sleep(0.05)  # Optional throttle to simulate real-time sending
                except Exception as e:
                    print(f"Error sending audio chunk: {e}")
                    break

            # Close WebSocket connection
            await websocket.close()
            print("WebSocket connection closed.")

# Run the test
asyncio.run(test_websocket())
