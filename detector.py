import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import deque

pTime = 0  # Previous Time (上一幀的時間)
cTime = 0  # Current Time (當前時間)

# ================= 設定區域 =================
# 1. 模型路徑
model_path = r"D:\project\posture_research\aug_experiment\weights\best.pt"

# 2. 攝影機 ID
camera_id = 1

# 3. 關鍵點索引 (P1-P4)
IDX_P1 = 0  # 頸椎 (Cervical) - (x1, y1)
IDX_P2 = 1  # 胸椎 (Thoracic) - (x2, y2)
IDX_P3 = 2  # 腰椎 (Lumbar)   - (x3, y3)
IDX_P4 = 3  # 骶椎 (Sacral)   - (x4, y4)

# 4. 正常範圍 (單位: 度)
# 格式: (最小值, 最大值, 平均值名稱)
REF_THETA1 = (80, 90, "Cervical")  # 頸椎段
REF_THETA2 = (70, 80, "Thoracic")  # 胸椎段
REF_THETA3 = (75, 85, "Lumbar")    # 腰椎段

# 5. 容許誤差 (閾值)
# 超過 (Min - 10) 或 (Max + 10) 即視為異常
TOLERANCE = 10 

# ===========================================

def calculate_inclination(p_curr, p_next):
    """
    計算脊柱段相對於水平面的傾斜角 (Inclination Angle)
    公式: theta = arctan((y_next - y_curr) / (x_next - x_curr))
    """
    x1, y1 = p_curr
    x2, y2 = p_next
    
    dy = y2 - y1
    dx = x2 - x1
    
    # 處理垂直線 (dx=0) 的情況，避免除以零錯誤
    if abs(dx) < 1e-6:
        return 90.0
    
    # 計算角度為度數
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    return angle_deg

def check_posture_status(angle, ref_range, tolerance):
    """
    比對角度是否在文獻範圍內
    """
    ref_min, ref_max, name = ref_range
    
    # 計算容許範圍
    thresh_min = ref_min - tolerance
    thresh_max = ref_max + tolerance   
    
    status = "Normal"
    color = (0, 255, 0) # 綠色
    
    if thresh_max < angle or angle< thresh_min:
        status = "Abnormal" # 不在正常範圍內
        color = (0, 0, 255) # 紅色

    
    return status, color, (thresh_max, thresh_max)

def run_analysis():
    print(f"🚀 載入模型: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    pTime = 0
    fps_buffer = deque(maxlen=30)
    print("✅ 系統啟動！按 'q' 退出。")

    while True:
        ret, frame = cap.read()
        if not ret: break


        # 執行預測
        results = model(frame, stream=True, conf=0.3, verbose=False,max_det=1)

        for result in results:
            if result.keypoints is None: continue
            
            # 獲取座標
            keypoints = result.keypoints.xy.cpu().numpy()

            for kpts in keypoints:
                # 確保 4 個點都偵測到
                if np.all(kpts > 0):
                    p1 = kpts[IDX_P1] # 頸
                    p2 = kpts[IDX_P2] # 胸
                    p3 = kpts[IDX_P3] # 腰
                    p4 = kpts[IDX_P4] # 骶

                    # --- 1. 計算三個傾斜角 (Thetas) ---
                    theta1 = calculate_inclination(p1, p2) # 頸椎段
                    theta2 = calculate_inclination(p2, p3) # 胸椎段
                    theta3 = calculate_inclination(p3, p4) # 腰椎段

                    # --- 2. 判斷狀態 ---
                    s1, c1, r1 = check_posture_status(theta1, REF_THETA1,TOLERANCE)
                    s2, c2, r2 = check_posture_status(theta2, REF_THETA2,TOLERANCE)
                    s3, c3, r3 = check_posture_status(theta3, REF_THETA3,TOLERANCE)

                    # --- 3. 繪圖 ---
                    # 畫連線
                    pts = np.array([p1, p2, p3, p4], np.int32)
                    cv2.polylines(frame, [pts], False, (200, 200, 200), 2)
                    
                    # 畫關鍵點
                    for p in pts:
                        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 0, 255), -1)

                    # --- 4. 顯示資訊面板 (畫在左側，避免遮擋人像) ---
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 標題
                    cv2.putText(frame, "Spine Segment Analysis", (20, 35), font, 0.7, (255, 255, 255), 2)
                    
                    # 顯示 Theta 1 (Cervical)
                    text1 = f"Th1(Cerv): {theta1:.1f} deg"
                    cv2.putText(frame, text1, (20, 70), font, 0.6, c1, 2)
                    cv2.putText(frame, f"Range: {r1[0]}-{r1[1]} ({s1})", (20, 90), font, 0.4, (200, 200, 200), 1)

                    # 顯示 Theta 2 (Thoracic)
                    text2 = f"Th2(Thor): {theta2:.1f} deg"
                    cv2.putText(frame, text2, (20, 120), font, 0.6, c2, 2)
                    cv2.putText(frame, f"Range: {r2[0]}-{r2[1]} ({s2})", (20, 140), font, 0.4, (200, 200, 200), 1)

                    # 顯示 Theta 3 (Lumbar)
                    text3 = f"Th3(Lumb): {theta3:.1f} deg"
                    cv2.putText(frame, text3, (20, 170), font, 0.6, c3, 2)
                    cv2.putText(frame, f"Range: {r3[0]}-{r3[1]} ({s3})", (20, 190), font, 0.4, (200, 200, 200), 1)

                    # 在身體旁邊也標註簡單數值
                    cv2.putText(frame, f"{theta1:.0f}", (int(p1[0])+10, int(p1[1])), font, 0.5, c1, 2)
                    cv2.putText(frame, f"{theta2:.0f}", (int(p2[0])+10, int(p2[1])), font, 0.5, c2, 2)
                    cv2.putText(frame, f"{theta3:.0f}", (int(p3[0])+10, int(p3[1])), font, 0.5, c3, 2)

                    # --- 繪製半透明黑框 ---
                    # 1. 複製目前的畫面做為圖層 (Overlay)
                    overlay = frame.copy()
                    
                    # 2. 在圖層上畫實心黑框
                    cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
                    
                    # 3. 設定透明度 (0.0 完全透明 ~ 1.0 完全不透明)
                    alpha = 0.4  # 建議值：0.3 ~ 0.5 看起來最舒服
                    
                    # 4. 混合圖片： frame = alpha * overlay + (1 - alpha) * frame
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        h, w, c = frame.shape
        # 計算 FPS (平滑化版)
        cTime = time.time()
        if cTime != pTime:
            current_fps = 1 / (cTime - pTime)
        else:
            current_fps = 0
        pTime = cTime
        fps_buffer.append(current_fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        # 準備要顯示的文字
        fps_text = f"FPS: {int(avg_fps)}"
        pos_x = w - 160 
        pos_y = 50      

        #顯示文字
        cv2.putText(frame, fps_text, (pos_x, pos_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        cv2.imshow('Spine Segment Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_analysis()