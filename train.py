from ultralytics import YOLO

# Windows 必須加這行，否則多工處理會崩潰
if __name__ == '__main__':
    
    # 1. 載入模型 (建議用 Pose 專用版)
    model = YOLO('yolo11s-pose.pt') 

    # 2. 開始訓練
    results = model.train(
        # 你的設定
        data=r'D:\project\data.yaml',
        task='pose',
        epochs=150,
        imgsz=640,
        batch=16,
        project='posture_research',
        name='aug_experiment',
        
        # 增強參數
        mosaic=1.0, 
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        translate=0.1,
        
        # 系統設定
        device=0,
        patience=30,
        verbose=True,
        workers=4 # Windows 上設太大有時會卡住，建議 2~4 即可
    )