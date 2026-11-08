# 作業二 主題 B：Bare PCB OpenCV（Adaptive Threshold blockSize）

> 機器視覺與 AI 演算法應用｜作業二。原作者的說明保留在 [README_原作者.md](README_原作者.md)。

## 1. 這份作業做了什麼

用影像相減找出裸 PCB 與無缺陷模板的差異（模板二值圖 − 測試圖二值圖），再數輪廓。
作業要求修改 **`cv2.adaptiveThreshold` 的 `blockSize`**，做三組以上實驗並記錄「差異遮罩前景比例」與「輪廓數」。

| 設定 | blockSize | 說明 |
|---|---|---|
| 1 | **15** | 原作者設定（原始） |
| 2 | 11 | 老師指定 |
| 3 | 21 | 老師指定 |
| 4 | 31 | 老師指定 |

只改 `blockSize`，其餘（C=5、resize 750×450、高斯模糊 3×3、中值濾波 5×5、面積門檻 300）全部固定。

> 資料集沒有缺陷位置標註，所以前景比例與輪廓數只是**觀察指標**，不是檢出率或準確率，也不計算 Precision／Recall／F1。

## 2. 環境需求

- Python 3.12（作者測試版本：3.12.6，Windows 11；程式用 `pathlib`，macOS／Linux 預期可執行，但未實測）
- 套件：`opencv-python`、`numpy`、`matplotlib`（測試版本：opencv-python 5.0.0、numpy 2.5.3、matplotlib 3.11.2）
- 只有要執行 `.ipynb` 時才需要 `ipykernel`

```bash
pip install -r requirements.txt
```

## 3. 資料夾內容

```text
Bare-PCB-defect-detection-using-OpenCV/
├── Bare-PCB-defect-detection-using-opencv.ipynb   原作者 notebook（修改前；只修正資料夾路徑，檔內另含執行後留下的輸出）
├── pcb_experiment.py     本作業主程式（修改後）：blockSize 批次實驗
├── pcb_noise_check.py    原因查證用的診斷程式（不屬於三組實驗）
├── Bare-PCB-images-dataset/   資料集（5 塊板的模板、三類缺陷圖，含旋轉版本）
├── results/              執行後自動產生的結果（見下）
├── requirements.txt
├── README.md             本檔
└── README_原作者.md       原作者的 README
```

## 4. 如何重現結果

**請先把終端機移到本資料夾**（所有路徑都由程式檔位置推算，不依賴你放在哪裡，也不受中文路徑影響）：

```bash
cd Bare-PCB-defect-detection-using-OpenCV
```

### 4-1 三組實驗（約 1 秒）

```bash
python pcb_experiment.py
```

對 PCB 01 的三張測試圖（缺孔 `01_missing_hole_10`、鼠咬 `01_mouse_bite_01`、斷路 `01_open_circuit_01`），
各以 blockSize 15／11／21／31 跑一次。輸出：

- `results/blockSize_15/`、`blockSize_11/`、`blockSize_21/`、`blockSize_31/`：模板與測試圖二值圖、差異遮罩（PNG）
- `results/results.csv`：每張圖每組的 `fg_ratio`（前景比例）、`contours_all`（全部輪廓數）、`blobs_lt300`（面積小於 300 的輪廓數，即原作者定義的缺陷數）

### 4-2 擴大測試圖（約 25 秒）

```bash
python pcb_experiment.py --full
```

用 5 塊板（01、05、07、10、11）所有**未旋轉**的缺陷圖共 166 張重跑（Missing_hole 55、Mouse_bite 55、Open_circuit 56，各自對應同編號模板）。
輸出 `results/results_full.csv`，並在終端機印出平均值與統計。

### 4-3 原因查證（約 2 秒，可選）

```bash
python pcb_noise_check.py
```

檢查「前景是不是雜訊」：輸出 `results/noise_check.csv`。

### 4-4 執行原作者 notebook（修改前）

用 VSCode 或 Jupyter 開 `Bare-PCB-defect-detection-using-opencv.ipynb`，核心選有裝 `opencv-python`、`matplotlib`、`ipykernel` 的 Python，**由上而下全部執行**。
最後一個 cell 應印出：

```text
Number of defects in image:
3
```

> notebook 用相對路徑讀圖，所以工作目錄必須是本資料夾（VSCode 直接開本資料夾即可）。

## 5. 預期結果（用來核對你重現得對不對）

### 5-1 `results/results.csv`（3 張 PCB 01 測試圖）

| blockSize | missing_hole_10 | mouse_bite_01 | open_circuit_01 |
|---|---|---|---|
| 15（原始） | 0.000121 / 3 / 3 | 0.000000 / 0 / 0 | 0.000163 / 2 / 2 |
| 11 | 0.000077 / 3 / 2 | 0.000127 / 3 / 2 | 0.000083 / 1 / 1 |
| 21 | 0.000133 / 3 / 3 | 0.000136 / 2 / 2 | 0.000293 / 3 / 3 |
| 31 | 0.000154 / 3 / 3 | 0.000299 / 3 / 3 | 0.000273 / 3 / 3 |

（每格依序為 前景比例 / 全部輪廓數 / 面積小於 300 的輪廓數）

### 5-2 `results/results_full.csv`（166 張，平均每張）

| blockSize | 前景比例 | 全部輪廓數 | 面積小於 300 的輪廓數 |
|---|---|---|---|
| 15（原始） | 0.0279% | 3.42 | 3.31 |
| 11 | 0.0217% | 3.40 | 2.96 |
| 21 | 0.0324% | 3.65 | 3.54 |
| 31 | 0.0364% | 4.81 | 4.41 |

重現驗證：在一份全新複本上（只複製程式與 notebook、資料集以資料夾連結共用）依序執行上述三支程式，
產生的 `results.csv`、`results_full.csv`、`noise_check.csv` 與作者原本的結果**逐位元組相同**（程式沒有隨機成分；`pcb_noise_check.py` 的人造雜訊使用固定種子）。

## 6. 程式修改摘要（修改前 → 修改後）

| 項目 | 修改前（原 notebook） | 修改後（`pcb_experiment.py`） |
|---|---|---|
| 路徑 | `Bare-PCB-dataset/...`（資料夾不存在，`cv2.imread` 回傳 `None`） | `Bare-PCB-images-dataset`，依程式檔位置組路徑；讀不到圖直接報錯 |
| 預處理 | 模板與測試圖各寫一次相同步驟 | 包成 `preprocess(path, block_size)` |
| blockSize | 寫死 15（cell 7、10 各一處） | `BLOCK_SIZES = [15, 11, 21, 31]` 逐組執行 |
| 指標 | 只印面積 0～300 的輪廓數 | `measure()` 另算前景比例與全部輪廓數，存成 CSV |
| 輸出 | 只用 `plt.imshow` 顯示 | 存成 PNG 與 CSV |
| 測試圖 | 1 張 | 新增 `--full`，166 張 |

## 7. 常見問題

- **`FileNotFoundError: 讀不到圖片`**：沒有在本資料夾執行，或 `Bare-PCB-images-dataset/` 不完整。
- **notebook 選不到核心**：先 `pip install ipykernel`，重新載入 VSCode 後再選 Python。
- **想換成別的圖**：改 `pcb_experiment.py` 開頭的 `TEMPLATE_PATH`、`TEST_IMAGES`（其他參數不要動，才能和作業結果比較）。
