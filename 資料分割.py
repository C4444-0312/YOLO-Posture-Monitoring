import os
import shutil
import random

# 設定路徑
source_dir = r"C:\Users\User\OneDrive\桌面\project\archive\all_data" # 假設你把所有圖片和標籤先放在這裡
output_dir = r"C:\Users\User\OneDrive\桌面\project\dataset"

# 建立目標資料夾結構
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# 獲取所有圖片檔案 (假設是 .jpg, 也可以加 .png)
image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
random.shuffle(image_files) #隨機打亂

# 計算切分數量
total = len(image_files)
train_count = int(total * 0.8)
val_count = int(total * 0.1)
# 剩下的給 test

# 執行移動
for i, filename in enumerate(image_files):
    if i < train_count:
        split = 'train'
    elif i < train_count + val_count:
        split = 'val'
    else:
        split = 'test'
    
    # 移動圖片
    src_img = os.path.join(source_dir, filename)
    dst_img = os.path.join(output_dir, 'images', split, filename)
    shutil.copy(src_img, dst_img)
    
    # 移動對應的標籤 (假設是 .txt)
    label_file = filename.replace('.jpg', '.txt')
    src_label = os.path.join(source_dir, label_file)
    dst_label = os.path.join(output_dir, 'labels', split, label_file)
    
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)

print(f"資料切分完成！總數: {total}")