import os
import cv2

image_folder = "../shelf_dataset"
annotations_file = "../annotation.csv"
output_folder = "../shelf_dataset_adapted"

os.makedirs(output_folder, exist_ok=True)

with open(annotations_file, "r") as f:
  annotations = f.readlines()

# Pular o cabeçalho do csv
annotations = annotations[1:]

for idx, line in enumerate(annotations):
  filename, shelf_coords_str = line.strip().split(";")
  shelf_coords = list(map(float, shelf_coords_str.split(",")))

  image_path = os.path.join(image_folder, filename)
  img = cv2.imread(image_path)
  if img is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    continue

  height, width, _ = img.shape

  lines = []
  for y_normalized in shelf_coords:
    y = int(y_normalized * height)
    x1, y1 = 0, y
    x2, y2 = width, y
    lines.append((x1, y1, x2, y2))

  output_image_name = f"{idx + 1}.jpg"
  output_txt_name = f"{idx + 1}.txt"

  cv2.imwrite(os.path.join(output_folder, output_image_name), img)
  output_txt_path = os.path.join(output_folder, output_txt_name)
  
  with open(output_txt_path, "w") as txt_file:
    all_coords = [coord for line in lines for coord in line]
    txt_file.write(f"{len(lines)} " + " ".join(map(str, all_coords)) + "\n")

print("Conversão concluída!")
