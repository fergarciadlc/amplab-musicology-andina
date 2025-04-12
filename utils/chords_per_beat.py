import pandas as pd

def cargar_acordes(path_csv):
    return pd.read_csv(path_csv, sep=';', quotechar='"')

def cargar_beats(path_txt):
    with open(path_txt, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def alinear_acordes_a_beats(df_chords, beats):
    beat_chords = []

    # Añadir segmento inicial desde 0 hasta el primer beat
    if beats:
        beat_chords.append({'Start': 0.0, 'End': beats[0], 'Label': 'N'})

    # Asignar acordes a intervalos de beats
    for i in range(len(beats) - 1):
        start = beats[i]
        end = beats[i + 1]
        label_row = df_chords[(df_chords['Start'] <= start) & (df_chords['End'] > start)]
        label = label_row.iloc[0]['Label'] if not label_row.empty else 'N'
        beat_chords.append({'Start': start, 'End': end, 'Label': label})

    return pd.DataFrame(beat_chords)

def guardar_csv(df, output_path):
    df.to_csv(output_path, sep=';', index=False, quoting=1)  # quoting=1 para usar comillas

# --- USO ---
# Cambia estos paths por los tuyos
csv_acordes = '/home/cepatinog/ethnomusic/amplab-musicology-andina/annotations/if_0172.csv'
txt_beats = '/home/cepatinog/ethnomusic/amplab-musicology-andina/data/Beat_annotations/if_0172.txt'
salida = 'beats/acordes_por_beat.csv'

# Proceso
df_chords = cargar_acordes(csv_acordes)
beats = cargar_beats(txt_beats)
df_resultado = alinear_acordes_a_beats(df_chords, beats)
guardar_csv(df_resultado, salida)

print(f"Acordes por beat guardados en: {salida}")
