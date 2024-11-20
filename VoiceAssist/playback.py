import sounddevice as sd
import soundfile as sf
import sys

def play_wav(filename):
    # Read the wave file
    print(f"Playing {filename}")
    data, samplerate = sf.read(filename)
    print(f"Sample rate: {samplerate} Hz")
    print(f"Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
    print(f"Length: {len(data)/samplerate:.2f} seconds")
    
    # Play the file
    sd.play(data, samplerate)
    sd.wait()  # Wait until file is done playing

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "debug_audio.wav"  # default file
    
    play_wav(filename)