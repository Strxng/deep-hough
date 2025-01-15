file_path = "NKL_file_list.txt"

with open(file_path, "w") as file:
  for i in range(22652, 22746):
    file.write(f"{i}\n")
    # file.write(f"NKL_resize_100_100/{i}\n")
    # file.write(f"NKL_resize_100_100/{i}_flip\n")

print(f"Arquivo salvo em: {file_path}")
