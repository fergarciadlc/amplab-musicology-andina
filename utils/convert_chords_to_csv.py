import os
import csv

input_dir = "data/Transcriptions"
output_dir = "data/annotations"

# Asegura que la carpeta de salida exista
os.makedirs(output_dir, exist_ok=True)

def process_chords_file(txt_path, output_name):
    with open(txt_path, "r") as f:
        lines = [line.strip().split('\t') for line in f if line.strip()]

    # Parse lines into (start, label)
    chords = [(float(start), label) for start, label in lines]

    # Calcular los intervalos start-end
    csv_rows = []
    for i in range(len(chords) - 1):
        start, label = chords[i]
        end = chords[i + 1][0]
        if label != "N":
            csv_rows.append((f"{start}", f"{end}", label))

    # Guardar CSV en carpeta annotations
    out_path = os.path.join(output_dir, f"{output_name}.csv")
    with open(out_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writerow(["Start", "End", "Label"])
        writer.writerows(csv_rows)
    print(f"Guardado: {out_path}")

def main():
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_chords.txt"):
                txt_path = os.path.join(root, file)
                output_name = file.replace("_chords.txt", "")
                process_chords_file(txt_path, output_name)

if __name__ == "__main__":
    main()
