import csv
import os

input_csv = "/home/cepatinog/ethnomusic/amplab-musicology-andina/beats/acordes_por_beat.csv"
output_dir = "/home/cepatinog/ethnomusic/amplab-musicology-andina/beats"

os.makedirs(output_dir, exist_ok=True)

def split_csv_by_row(input_csv, output_dir):
    with open(input_csv, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for idx, row in enumerate(reader, start=1):
            beat_name = f"beat_{idx:03d}.csv"
            output_path = os.path.join(output_dir, beat_name)

            with open(output_path, "w", newline='') as out_f:
                writer = csv.writer(out_f, delimiter=';', quoting=csv.QUOTE_ALL)
                writer.writerow(["Start", "End", "Label"])
                writer.writerow([row["Start"], row["End"], row["Label"]])

            print(f"Guardado: {output_path}")

if __name__ == "__main__":
    split_csv_by_row(input_csv, output_dir)
