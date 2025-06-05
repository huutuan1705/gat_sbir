import os
import pandas as pd

# Đường dẫn tới thư mục chứa ảnh
image_folder = 'D:/Research/GAT_SBIR/dataset/ShoeV2/photo'  # ← Sửa đường dẫn này cho đúng

# Lọc danh sách file là ảnh (đuôi .jpg, .png, .jpeg,...)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

# Ghi vào DataFrame
df = pd.DataFrame(image_files, columns=['image_name'])

# Ghi vào Excel
output_excel = 'D:/Research/GAT_SBIR/dataset/ShoeV2/ShoeV2_labels.xlsx'  # Tên file Excel đầu ra
df.to_excel(output_excel, index=False)

print(f"Đã ghi {len(image_files)} tên ảnh vào file '{output_excel}'")
