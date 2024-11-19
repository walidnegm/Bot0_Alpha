from pydub import AudioSegment
import simpleaudio as sa

# Load the audio file
audio_file = "debug_audio.wav"

# Convert to 16kHz mono if necessary
def convert_audio(input_file, output_file):
    sound = AudioSegment.from_wav(input_file)
    sound = sound.set_channels(1)  # Mono audio
    sound = sound.set_frame_rate(16000)  # 16kHz sample rate
    sound.export(output_file, format="wav")
    print(f"Audio converted and saved as {output_file}")

# Convert audio if needed
converted_file = "test_audio_16kHz_mono.wav"
convert_audio(audio_file, converted_file)

# Load the converted audio file
audio = AudioSegment.from_wav(converted_file)

# Play the audio using simpleaudio
def play_audio(audio_file):
    # Play the audio
    wave_obj = sa.WaveObject.from_wave_file(audio_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait for the audio to finish playing

play_audio(converted_file)
