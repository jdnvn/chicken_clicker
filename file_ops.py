import os

def generate_neg_txt():
  with open('neg.txt', 'w') as f:
    for filename in os.listdir('negative'):
      f.write(f'negative/{filename}\n')

def move_annotated():
  with open('pos_4.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
      path = line.split()[0]
      filename = path.split('/')[-1]
      os.rename(path, f"annotated/{filename}")

generate_neg_txt()
