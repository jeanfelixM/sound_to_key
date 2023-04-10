import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

def display_frames_and_spectrograms(key_n_frames, key_n_sounds, sample_rate=44100, columns=4):
    plt.figure(figsize=(15, 15))

    for i, key_frame in enumerate(key_n_frames, start=1):
        plt.subplot(2 * (len(key_n_frames) // columns + 1), columns, 2 * i - 1)
        plt.imshow(cv2.threshold(cv2.cvtColor(key_frame["frame"], cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1])
        plt.title(f"Key: {key_frame['key']}")
        plt.axis('off')

        # Trouver le son associé
        for key_n_sound in key_n_sounds:
            if key_n_sound["key"] == key_frame["key"]:
                audio_segment = key_n_sound["sound"]
                break

        # Générer le spectrogramme
        spectrogram = np.abs(librosa.stft(audio_segment))

        # Afficher le spectrogramme
        plt.subplot(2 * (len(key_n_frames) // columns + 1), columns, 2 * i)
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    plt.show()
