import torch
import pandas as pd

annotations_file = "ChairV2_labels.csv"
num_classes = 19
image_name = "00140128_v1.png"
img_labels_df = pd.read_csv(annotations_file)
labels_row = img_labels_df[img_labels_df["image_name"] == image_name]

# Kiểm tra nếu không tìm thấy
if labels_row.empty:
    raise ValueError(f"Image '{image_name}' not found in CSV file.")

# Lấy nhãn dưới dạng tensor
labels = labels_row.iloc[0, -num_classes:].values
labels = torch.tensor(labels.astype('float32'))

print(labels)