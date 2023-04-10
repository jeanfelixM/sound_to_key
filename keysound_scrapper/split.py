import ffmpeg

input_file = "vid.mp4"
output_audio = "audio_output.wav"
output_video = "video_output.mp4"

def extract_audio(input_file, output_audio):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_audio, vn=True, acodec="pcm_s16le", ar=44100, ac=1)
            .run()
        )
        print(f"Audio extrait avec succès: {output_audio}")
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'audio : {e}")

def extract_video(input_file, output_video):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_video, an=True, c='copy')
            .run()
        )
        print(f"Vidéo extraite avec succès: {output_video}")
    except Exception as e:
        print(f"Erreur lors de l'extraction de la vidéo : {e}")

if __name__ == "__main__":
    extract_audio(input_file, output_audio)
    extract_video(input_file, output_video)
