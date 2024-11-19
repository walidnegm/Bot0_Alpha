import asyncio
import websockets
import wave
import pydub  # To convert the audio to the correct format if needed

async def test_websocket():
    uri = "ws://127.0.0.1:8000/listen/"
    
    # Convert the audio to 16kHz, Mono (if it's not already)
    audio_file = "testfile.wav"
    converted_file = "test_audio_16kHz_mono.wav"
    
    # Check if the audio is already in the correct format (16kHz, mono)
    with wave.open(audio_file, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        print(f"Original Audio - Frame Rate: {frame_rate}, Channels: {channels}, Sample Width: {sample_width}")

    ## If the audio is not in the required format, convert it
    #if frame_rate != 16000 or channels != 1:
    #print("Audio format is not correct, converting to 16kHz mono...")
    sound = pydub.AudioSegment.from_wav(audio_file)
    #sound = sound.set_channels(1)  # Mono
    sound = sound.set_frame_rate(16000)  # 16kHz
    sound.export(converted_file, format="wav")
    audio_file = converted_file  # Update the file to the converted one
    
    # Open the converted (or original) audio file
    with wave.open(audio_file, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        print(f"Using Audio - Frame Rate: {frame_rate}, Channels: {channels}, Sample Width: {sample_width}")
        
        async with websockets.connect(uri) as websocket:
            # Send audio data in chunks
            chunk_size = 1024  # Smaller chunks for WebSocket
            while chunk := wav_file.readframes(chunk_size):
                try:
                    print(f"Sending {len(chunk)} bytes of audio...")
                    await websocket.send(chunk)  # Send raw audio bytes
                    await asyncio.sleep(0.1)  # Throttle the sending of audio data
                except Exception as e:
                    print(f"Error sending audio chunk: {e}")
                    break  # Exit if there is an error sending data
            
            # Optionally, close the connection
            await websocket.close()
            print("WebSocket connection closed.")

# Run the test
asyncio.run(test_websocket())
