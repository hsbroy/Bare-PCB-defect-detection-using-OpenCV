"""
主題 B：Bare PCB OpenCV —— Adaptive Threshold blockSize 實驗

流程（沿用原作者 notebook，只把 blockSize 變成參數）：
  讀灰階 → resize(750,450) → GaussianBlur(3,3) → adaptiveThreshold(MEAN_C, C=5)
  → 模板 − 測試（cv2.subtract）→ medianBlur(5) → findContours，面積 0~300 視為缺陷

固定條件：同一張模板、同一批測試圖、C=5、resize、模糊核、median 核、面積門檻都不變，
只改 blockSize（15=原作者設定，11／21／31=老師指定三組）。

用法：
  python pcb_experiment.py          三張 PCB 01 測試圖 × 四個 blockSize，存圖與 results/results.csv
  python pcb_experiment.py --full   擴大測試圖（166 張），存 results/results_full.csv

注意：資料集沒有缺陷標註，所以只記觀察指標（前景比例、輪廓數），
不可當成檢出率，也不計算 Precision／Recall／F1。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------- 固定條件（除 BLOCK_SIZES 外都不要改）----------
ROOT = Path(__file__).resolve().parent  # 這個程式檔所在的資料夾Bare-PCB-defect-detection-using-OpenCV
DATA = ROOT / "Bare-PCB-images-dataset" # 資料集根目錄Bare-PCB-defect-detection-using-OpenCV\Bare-PCB-images-dataset
RESULT_DIR = ROOT / "results"           # 實驗結果輸出資料夾Bare-PCB-defect-detection-using-OpenCV\results

TEMPLATE_PATH = DATA / "Template-PCB-images" / "01.JPG" # 模板圖路徑，固定
TEST_IMAGES = {  # 名稱 -> 路徑，皆為 PCB 01，對應同一張模板
    "missing_hole_10": DATA / "Defective-PCB-images" / "Missing_hole" / "Missing_hole_01" / "01_missing_hole_10.jpg",
    "mouse_bite_01": DATA / "Defective-PCB-images" / "Mouse_bite" / "Mouse_bite_01" / "01_mouse_bite_01.jpg",
    "open_circuit_01": DATA / "Defective-PCB-images" / "Open_circuit" / "Open_circuit_01" / "01_open_circuit_01.jpg",
}

BLOCK_SIZES = [15, 11, 21, 31]  # 15 = 原始設定；blockSize 必須是大於 1 的奇數
C = 5                           # adaptiveThreshold 的常數，固定
SIZE = (750, 450)               # resize 大小 (寬, 高)，固定
BLUR_KERNEL = (3, 3)            # 高斯模糊核，固定
MEDIAN_KERNEL = 5               # 差異圖去雜訊的中值濾波核，固定
MAX_BLOB_AREA = 300             # 原作者：面積 0~300 的輪廓算一個缺陷


# ---------- 讀寫圖（路徑含中文時 cv2.imread／imwrite 在 Windows 可能失敗，所以改用 numpy 轉手）----------
def read_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:  # 讀不到就直接報錯，避免之後才冒出看不懂的錯誤
        raise FileNotFoundError(f"讀不到圖片：{path}")
    return img


def save_png(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"存檔失敗：{path}")
    buf.tofile(str(path))


# ---------- 修改處 1：把 notebook cell 4~7、9~10 的重複步驟包成函式 ----------
def preprocess(path, block_size):
    """讀灰階 → resize → 高斯模糊 → adaptiveThreshold，回傳二值圖。"""
    img = read_gray(path)
    img = cv2.resize(img, SIZE)
    img = cv2.GaussianBlur(img, BLUR_KERNEL, 0)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, C)


# ---------- 對應 notebook cell 11~12 ----------
def diff_mask(template_bin, test_bin):
    """模板 − 測試，再用中值濾波去背景雜訊。回傳 (相減結果, 最終遮罩)。"""
    sub_img = cv2.subtract(template_bin, test_bin)
    final_img = cv2.medianBlur(sub_img, MEDIAN_KERNEL)
    return sub_img, final_img


# ---------- 修改處 2：新增老師要的兩個指標 ----------
def measure(final_img):
    """回傳 (前景比例, 全部輪廓數, 面積 0~300 的輪廓數=原作者的缺陷數)。"""
    fg_ratio = float(np.count_nonzero(final_img)) / final_img.size
    cnts = cv2.findContours(final_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    n_all = len(cnts)
    n_blobs = sum(1 for c in cnts if 0 < cv2.contourArea(c) < MAX_BLOB_AREA)
    return fg_ratio, n_all, n_blobs


# ---------- 修改處 3：對每個 blockSize、每張測試圖重複跑同一流程 ----------
def run_experiments():
    rows = []
    for bs in BLOCK_SIZES:
        template_bin = preprocess(TEMPLATE_PATH, bs)
        out_dir = RESULT_DIR / f"blockSize_{bs}"
        save_png(out_dir / "template_bin.png", template_bin)
        for name, path in TEST_IMAGES.items():
            test_bin = preprocess(path, bs)
            sub_img, final_img = diff_mask(template_bin, test_bin)
            fg_ratio, n_all, n_blobs = measure(final_img)
            save_png(out_dir / f"{name}_test_bin.png", test_bin)
            save_png(out_dir / f"{name}_final.png", final_img)
            rows.append({"blockSize": bs, "image": name,
                         "fg_ratio": round(fg_ratio, 6),
                         "contours_all": n_all, "blobs_lt300": n_blobs})
            print(f"blockSize={bs:>2}  {name:<16} 前景比例={fg_ratio:.4%}  "
                  f"全部輪廓={n_all}  面積<300={n_blobs}")

    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已輸出：{RESULT_DIR / 'results.csv'}")


# ---------- 修改處 4：擴大測試圖（python pcb_experiment.py --full）----------
DEFECTS = ["Missing_hole", "Mouse_bite", "Open_circuit"]
BOARDS = ["01", "05", "07", "10", "11"]


def collect_full_set():
    """選圖規則（事先固定、不挑圖）：5 個板號所有未旋轉的缺陷圖，模板為同編號的 <板號>.JPG。
    回傳 [(缺陷, 板號, 檔名, 路徑)]。"""
    items = []
    for d in DEFECTS:
        for b in BOARDS:
            folder = DATA / "Defective-PCB-images" / d / f"{d}_{b}"
            for p in sorted(folder.glob("*.jpg")):
                items.append((d, b, p.name, p))
    return items


def run_full():
    """用全部未旋轉缺陷圖（166 張）對四個 blockSize 重跑同一流程，只存統計、不存圖。"""
    items = collect_full_set()
    print(f"測試圖共 {len(items)} 張")
    tpl_cache, rows = {}, []
    for d, b, name, path in items:
        tpl_path = DATA / "Template-PCB-images" / f"{b}.JPG"
        raw_t = cv2.resize(read_gray(tpl_path), SIZE).astype(np.float32)
        raw_x = cv2.resize(read_gray(path), SIZE).astype(np.float32)
        shift_len = float(np.hypot(*cv2.phaseCorrelate(raw_t, raw_x)[0]))  # 模板與測試圖的位移檢查
        for bs in BLOCK_SIZES:
            if (b, bs) not in tpl_cache:
                tpl_cache[(b, bs)] = preprocess(tpl_path, bs)
            _, final_img = diff_mask(tpl_cache[(b, bs)], preprocess(path, bs))
            fg, n_all, n_blob = measure(final_img)
            rows.append({"defect": d, "board": b, "image": name, "blockSize": bs,
                         "fg_ratio": fg, "contours_all": n_all, "blobs_lt300": n_blob,
                         "shift_px": round(shift_len, 3)})

    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / "results_full.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def summarize(title, keyfunc):
        print(f"\n== {title} ==")
        groups = defaultdict(list)
        for r in rows:
            groups[(keyfunc(r), r["blockSize"])].append(r)
        for k in sorted({k for k, _ in groups}):
            line = f"{str(k):<14}"
            for bs in BLOCK_SIZES:
                g = groups[(k, bs)]
                line += (f" | bs{bs}: fg={np.mean([x['fg_ratio'] for x in g]):.4%} "
                         f"輪廓={np.mean([x['contours_all'] for x in g]):.2f} "
                         f"<300={np.mean([x['blobs_lt300'] for x in g]):.2f}")
            print(line)

    summarize("全部（平均每張）", lambda r: "ALL")
    summarize("依缺陷類型", lambda r: r["defect"])
    summarize("依板號", lambda r: r["board"])

    shifts = [r["shift_px"] for r in rows if r["blockSize"] == BLOCK_SIZES[0]]
    print(f"\n位移（像素）：最大 {max(shifts):.2f}，中位數 {np.median(shifts):.3f}，>1px 的圖 {sum(s > 1 for s in shifts)} 張")
    by_img = defaultdict(dict)
    for r in rows:
        by_img[(r["defect"], r["board"], r["image"])][r["blockSize"]] = r
    n = len(by_img)
    up_fg = sum(v[31]["fg_ratio"] > v[11]["fg_ratio"] for v in by_img.values())
    up_ct = sum(v[31]["contours_all"] > v[11]["contours_all"] for v in by_img.values())
    print(f"blockSize 31 對 11：前景比例變多 {up_fg}/{n} 張；全部輪廓數變多 {up_ct}/{n} 張")
    print("已輸出：", RESULT_DIR / "results_full.csv")


if __name__ == "__main__":
    if "--full" in sys.argv:
        run_full()
    else:
        run_experiments()
