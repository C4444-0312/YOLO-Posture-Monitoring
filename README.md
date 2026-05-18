# 🚀 YOLO11 即時脊椎姿態監測系統 (Real-Time Spine Posture Monitoring)

基於 YOLO11-pose 與電腦視覺技術開發的即時人體姿態分析系統。本系統不僅能精準偵測人體關鍵骨架，更能進一步計算**頸椎、胸椎、腰椎**的關節傾斜角度，並將其與正常生理學範圍進行交叉比對，即時給予姿勢異常警告。

#程式碼由GEMINI產生

## ✨ 核心亮點 (Key Features)

- **🤖 前沿 AI 落地應用**：採用最新的 `YOLO11s-pose` 模型進行人體姿態估計，兼具高準確度與即時推論速度 (FPS)。
- **📊 視覺化即時反饋**：使用 OpenCV 打造流暢的 UI 儀表板，即時顯示 FPS、各關節角度及健康狀態評估 (Normal / Abnormal)。

---
## 🔬 演算法與醫學評估邏輯 (Algorithm & Biomechanics)

本專案將人體脊椎簡化為四個核心關鍵點 (Keypoints)：
1. **P1 (Cervical / 頸椎)**
2. **P2 (Thoracic / 胸椎)**
3. **P3 (Lumbar / 腰椎)**
4. **P4 (Sacral / 骶椎)**

透過計算相鄰關鍵點的相對夾角 (Thetas)，並設定醫學參考範圍 (Reference Range) 與容許誤差 (Tolerance = ±10度) 進行評估：

| 脊椎段 (Segment) | 計算節點 | 正常角度範圍 (Ref. Range) | 異常判定標準 |
| :--- | :---: | :---: | :--- |
| **Theta 1 (頸椎段)** | P1 - P2 | 80° - 90° | < 70° 或 > 100° |
| **Theta 2 (胸椎段)** | P2 - P3 | 70° - 80° | < 60° 或 > 90° |
| **Theta 3 (腰椎段)** | P3 - P4 | 75° - 85° | < 65° 或 > 95° |

*(系統會根據上述邏輯，在畫面上自動以綠色 (Normal) 或紅色 (Abnormal) 標示當前姿勢狀態。)*

---

## 📂 資料集來源與前置處理 (Dataset & Preprocessing)

- **原始資料集**：Kaggle - [Posture Keypoints Detection](https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection)
- **資料處理流程**：
  1. `資料分割.py`: 將清洗後的資料集自動依 `80% Train`, `10% Val`, `10% Test` 比例劃分。

---

## 🏗️ 專案檔案結構 (Repository Structure)

```text
.
├── detector.py      # 即時推論與 UI 視覺化主程式 (OpenCV 整合)
├── train.py         # YOLO11-pose 模型訓練腳本 (包含資料增強設定)
├── 資料分割.py      # 訓練/驗證/測試集自動化拆分工具
├── data.yaml        # YOLO 訓練所需之資料集設定檔
└── posture_research/ # 訓練輸出結果 (包含 Loss 曲線、mAP 評估指標)
```

## 👨‍💻 我的角色與貢獻 (My Role & Contributions)

本專案為團隊協作成果，我於團隊中擔任**核心程式開發 (Core Developer)**，主要負責以下技術實作：
- **AI 模型與演算法**：撰寫 YOLO11 模型訓練腳本 (`train.py`)，並實作脊椎傾斜角度的數學計算與狀態判定邏輯 (`detector.py`)。
- **即時推論系統**：整合 OpenCV 串接 WebCam，開發具備高 FPS 且帶有數據視覺化反饋的即時分析儀表板。
