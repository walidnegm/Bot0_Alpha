import sounddevice as sd
import numpy as np

# Generate a simple tone
duration = 5.0  # seconds
samplerate = 44100
frequency = 440.0  # Hz
t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

sd.play(audio, samplerate)
sd.wait()
print("Sound played!")