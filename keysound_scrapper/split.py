import cv2
import ffmpeg
import os
import librosa

input_file = "vid.mp4"
output_audio = "audio_output.wav"
output_video = "video_output.mp4"

def extract_audio(input_file):
    try:
        output_audio = 'temp_audio.wav'
        (
            ffmpeg
            .input(input_file)
            .output(output_audio, vn=True, acodec="pcm_s16le", ar=44100, ac=1, format='wav')
            .run()
        )
        audio_data, sample_rate = librosa.load(output_audio, sr=None, mono=True)
        os.remove(output_audio)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Erreur lors de la séparation de l'audio : {e}")
        return None

