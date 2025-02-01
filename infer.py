import argparse
import os

import torch
import torch.optim
import numpy as np
import yaml
import cv2

from PIL import Image
from torchvision import transforms
from model.network import Net
from skimage.measure import label, regionprops

# Utils

def get_boundary_point(y, x, angle, H, W):
  '''
  Given point y,x with angle, return a two point in image boundary with shape [H, W]
  return point:[x, y]
  '''
  point1 = None
  point2 = None
  
  if angle == -np.pi / 2:
    point1 = (x, 0)
    point2 = (x, H-1)
  elif angle == 0.0:
    point1 = (0, y)
    point2 = (W-1, y)
  else:
    k = np.tan(angle)
    if y-k*x >=0 and y-k*x < H:  #left
      if point1 == None:
        point1 = (0, int(y-k*x))
      elif point2 == None:
        point2 = (0, int(y-k*x))
        if point2 == point1: point2 = None
    # print(point1, point2)
    if k*(W-1)+y-k*x >= 0 and k*(W-1)+y-k*x < H: #right
      if point1 == None:
        point1 = (W-1, int(k*(W-1)+y-k*x))
      elif point2 == None:
        point2 = (W-1, int(k*(W-1)+y-k*x)) 
        if point2 == point1: point2 = None
    # print(point1, point2)
    if x-y/k >= 0 and x-y/k < W: #top
      if point1 == None:
        point1 = (int(x-y/k), 0)
      elif point2 == None:
        point2 = (int(x-y/k), 0)
        if point2 == point1: point2 = None
    # print(point1, point2)
    if x-y/k+(H-1)/k >= 0 and x-y/k+(H-1)/k < W: #bottom
      if point1 == None:
        point1 = (int(x-y/k+(H-1)/k), H-1)
      elif point2 == None:
        point2 = (int(x-y/k+(H-1)/k), H-1)
        if point2 == point1: point2 = None
    # print(int(x-y/k+(H-1)/k), H-1)
    if point2 == None : point2 = point1
  return point1, point2

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
  #return type: [(y1, x1, y2, x2)]
  H, W = size
  irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
  itheta = np.pi / numAngle
  b_points = []

  for (thetai, ri) in point_list:
    theta = thetai * itheta
    r = ri - numRho // 2
    cosi = np.cos(theta) / irho
    sini = np.sin(theta) / irho
    if sini == 0:
      x = np.round(r / cosi + W / 2)
      b_points.append((0, int(x), H-1, int(x)))
    else:
      angle = np.arctan(- cosi / sini)
      y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
      p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
      if p1 is not None and p2 is not None:
        b_points.append((p1[1], p1[0], p2[1], p2[0]))
  return b_points

# def visulize_mapping(b_points, size, filename):
#   img = cv2.imread(os.path.join(filename)) #change the path when using other dataset.
#   img = cv2.resize(img, size)
#   for (y1, x1, y2, x2) in b_points:
#     img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=int(0.01*max(size[0], size[1])))
#   return img

# Argumentos

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
parser.add_argument('--model', required=True, help='path to the pretrained model')
parser.add_argument('--image', required=True, help='path to the pretrained model')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

CONFIGS = yaml.load(open("./config.yml"), Loader=yaml.SafeLoader)

# Configurando modelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
model = model.cuda(device=device)

checkpoint = torch.load(args.model)

if 'state_dict' in checkpoint.keys():
  model.load_state_dict(checkpoint['state_dict'])
else:
  model.load_state_dict(checkpoint)

model.eval()

# Transform da imagem

transform = transforms.Compose([
  transforms.Resize((400, 400)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inferencia

with torch.no_grad():
  # Carregar e pré-processar a imagem
  image = Image.open(args.image).convert('RGB')
  original_size = image.size[::-1]
  image_tensor = transform(image).unsqueeze(0).to(device)

  # Inferência
  key_points = model(image_tensor)
  key_points = torch.sigmoid(key_points)

  # Pós-processamento
  # visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
  # os.makedirs(visualize_save_path, exist_ok=True)

  binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
  kmap_label = label(binary_kmap, connectivity=1)
  props = regionprops(kmap_label)
  plist = [prop.centroid for prop in props]

  # Reverter mapeamento para coordenadas
  b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
  scale_w = original_size[1] / 400
  scale_h = original_size[0] / 400

  for i in range(len(b_points)):
    y1 = int(np.round(b_points[i][0] * scale_h))
    x1 = int(np.round(b_points[i][1] * scale_w))
    y2 = int(np.round(b_points[i][2] * scale_h))
    x2 = int(np.round(b_points[i][3] * scale_w))
    
    if x1 == x2:
      angle = -np.pi / 2
    else:
      angle = np.arctan((y1 - y2) / (x1 - x2))
    
    (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, original_size[0], original_size[1])
    b_points[i] = (y1, x1, y2, x2)

  # Visualizar e salvar imagem com as linhas
  # vis = visulize_mapping(b_points, original_size[::-1], args.image)
  # output_image_path = os.path.join(visualize_save_path, os.path.basename(args.image))
  # cv2.imwrite(output_image_path, vis)

  output_string = f"{len(b_points)} "
  output_string += ' '.join(' '.join(map(str, point)) for point in b_points)
  output_string += '\n'

  print(f"{output_string}")