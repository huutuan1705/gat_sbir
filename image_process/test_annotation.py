import torch
import pandas as pd

annotations_file = "ChairV2_labels.csv"
num_classes = 19
idx = 1
img_labels_df = pd.read_csv(annotations_file)
labels = img_labels_df.iloc[idx, -num_classes:].values
labels = torch.tensor(labels.astype('float32'))

print(labels)