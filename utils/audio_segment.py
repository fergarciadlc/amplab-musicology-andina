import os
from pydub import AudioSegment

# --- CONFIGURACIÓN ---
AUDIO_FILE = "/home/cepatinog/ethnomusic/amplab-musicology-andina/data/Audio/if_0172.wav"      # Ruta del archivo de audio
BEAT_FILE = "/home/cepatinog/ethnomusic/amplab-musicology-andina/data/Beat_annotations/if_0172.txt"                # Ruta del archivo de tiempos (en segundos)
OUTPUT_DIR = "beats"                   # Carpeta de salida

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar el audio
audio = AudioSegment.from_wav(AUDIO_FILE)

# Leer los tiempos de beats
with open(BEAT_FILE, "r") as f:
    beat_times = [float(line.strip()) for line in f if line.strip()]

# Convertir a milisegundos
beat_times_ms = [int(t * 1000) for t in beat_times]

# Agregar último tiempo igual a la duración total del audio si no está incluido
if beat_times_ms[-1] < len(audio):
    beat_times_ms.append(len(audio))

# Segmentar y exportar
for i in range(len(beat_times_ms) - 1):
    start = beat_times_ms[i]
    end = beat_times_ms[i + 1]
    segment = audio[start:end]
    segment.export(f"{OUTPUT_DIR}/beat_{i+1:03d}.wav", format="wav")

print(f"{len(beat_times_ms)-1} segmentos exportados a la carpeta '{OUTPUT_DIR}'.")
