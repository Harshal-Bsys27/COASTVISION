import os

labels_dir = 'dataset/train/labels'
counts = [0, 0, 0]
for fname in os.listdir(labels_dir):
    if fname.endswith('.txt'):
        with open(os.path.join(labels_dir, fname)) as f:
            for line in f:
                idx = int(line.split()[0])
                counts[idx] += 1
print(f"Drowning: {counts[0]}, Person out of water: {counts[1]}, Swimming: {counts[2]}")
