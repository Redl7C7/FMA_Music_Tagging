import os
import torchaudio

# Pfad zum Verzeichnis mit dem FMA-Datensatz
fma_full_dir = 'E:/Neuer Ordner/fma_full/'

# Funktion zum Suchen einer bestimmten Audiodatei im Dataset
def find_audio_file(root_dir, track_id):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp3') and file.startswith(f'{track_id:06d}'):
                return os.path.join(root, file)
    return None

# Funktion zum Konvertieren und Speichern der Audiodatei
def convert_and_save_audio(root_dir, track_id, output_dir):
    # Finde die Audiodatei im Dataset
    audio_file = find_audio_file(root_dir, track_id)
    if audio_file is None:
        print(f'Audio file for track ID {track_id} not found.')
        return

    # Lade die Audiodatei
    waveform, sample_rate = torchaudio.load(audio_file)

    # Konvertiere und speichere die Audiodatei
    output_file = os.path.join(output_dir, f'{track_id:06d}.wav')
    torchaudio.save(output_file, waveform, sample_rate)
    print(f'Audio file for track ID {track_id} saved to {output_file}.')

# Track ID des Songs, den du konvertieren möchtest
track_id = 12345

# Verzeichnis zum Speichern der konvertierten Audiodatei
output_dir = 'E:/Neuer Ordner/'

# Konvertiere und speichere den Song
convert_and_save_audio(fma_full_dir, track_id, output_dir)
