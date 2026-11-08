"""
原因查證（診斷用，非作業要求的三組實驗）
推論：差異遮罩的前景主要來自「近門檻像素被兩張照片間的微小雜訊翻轉」，
      blockSize 越大，近門檻像素越多，翻轉越多。

方法 1：檢查相減後的白色像素，是「小灰階差（雜訊等級）」還是「大灰階差（可能是真差異）」，
        以及是否集中在近門檻像素。
方法 2：人造對照——把模板加上已知大小的高斯雜訊當作「測試圖」（沒有任何缺陷），
        其餘流程不變；若前景仍隨 blockSize 增加，趨勢就是雜訊造成的。
        雜訊大小 σ 取自真實「模板 vs 測試圖」灰階差的穩健標準差，並加測 2 倍。
"""
import csv

import cv2
import numpy as np

import pcb_experiment as e

NEAR = 3          # |灰階 − 門檻| <= NEAR 視為近門檻
NOISE_MAX = 5     # 兩圖灰階差 <= 此值視為雜訊等級；>= STRONG 視為大差異
STRONG = 10
SEEDS = range(5)  # 方法 2 重複 5 次取平均（固定種子，可重現）


def blurred_gray(path_or_img):
    img = e.read_gray(path_or_img) if not isinstance(path_or_img, np.ndarray) else path_or_img
    img = cv2.resize(img, e.SIZE)
    return cv2.GaussianBlur(img, e.BLUR_KERNEL, 0)


def binarize(g, bs):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, bs, e.C)


def threshold_map(g, bs):
    return cv2.blur(g.astype(np.float32), (bs, bs), borderType=cv2.BORDER_REPLICATE) - e.C


def method1(rows):
    print("== 方法 1：相減後白色像素的來源（真實三張測試圖）==")
    tpl = blurred_gray(e.TEMPLATE_PATH)
    for bs in e.BLOCK_SIZES:
        tb = binarize(tpl, bs)
        thr_t = threshold_map(tpl, bs)
        for name, path in e.TEST_IMAGES.items():
            tst = blurred_gray(path)
            sub, final = e.diff_mask(tb, binarize(tst, bs))
            diff = np.abs(tpl.astype(np.int16) - tst.astype(np.int16))
            near = (np.abs(tpl - thr_t) <= NEAR) | (np.abs(tst - threshold_map(tst, bs)) <= NEAR)
            for label, mask in (("sub(濾波前)", sub > 0), ("final(濾波後)", final > 0)):
                n = int(mask.sum())
                if n == 0:
                    continue
                r = {"method": 1, "blockSize": bs, "image": name, "mask": label, "n_white": n,
                     "share_noise_level": round(float((diff[mask] <= NOISE_MAX).mean()), 4),
                     "share_strong": round(float((diff[mask] >= STRONG).mean()), 4),
                     "share_near_thr": round(float(near[mask].mean()), 4)}
                rows.append(r)
                print(f"bs={bs:>2} {name:<16}{label:<12} 白像素={n:>5}  "
                      f"灰階差<={NOISE_MAX}:{r['share_noise_level']:.1%}  "
                      f">={STRONG}:{r['share_strong']:.1%}  近門檻:{r['share_near_thr']:.1%}")


def method2(rows):
    print("\n== 方法 2：模板 + 人造高斯雜訊（無缺陷）==")
    tpl_raw = cv2.resize(e.read_gray(e.TEMPLATE_PATH), e.SIZE)
    # 真實差異的穩健標準差（MAD），代表兩張照片間的雜訊等級
    sigmas = []
    for path in e.TEST_IMAGES.values():
        d = (tpl_raw.astype(np.float32) - cv2.resize(e.read_gray(path), e.SIZE).astype(np.float32)).ravel()
        sigmas.append(1.4826 * np.median(np.abs(d - np.median(d))))
    sigma_real = float(np.mean(sigmas))
    print(f"真實差異的穩健標準差 σ_real ≈ {sigma_real:.2f} 灰階")

    for label, sigma in (("0（對照：完全相同）", 0.0), ("1×σ_real", sigma_real), ("2×σ_real", 2 * sigma_real)):
        print(f"\n雜訊 {label}，σ={sigma:.2f}")
        for bs in e.BLOCK_SIZES:
            fg, n_all, n_blob = [], [], []
            tb = binarize(blurred_gray(tpl_raw), bs)
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                noisy = np.clip(tpl_raw + rng.normal(0, sigma, tpl_raw.shape), 0, 255).round().astype(np.uint8)
                _, final = e.diff_mask(tb, binarize(blurred_gray(noisy), bs))
                f, a, b = e.measure(final)
                fg.append(f); n_all.append(a); n_blob.append(b)
            r = {"method": 2, "noise": label, "sigma": round(sigma, 3), "blockSize": bs,
                 "fg_ratio_mean": round(float(np.mean(fg)), 6),
                 "contours_all_mean": float(np.mean(n_all)), "blobs_lt300_mean": float(np.mean(n_blob))}
            rows.append(r)
            print(f"  bs={bs:>2}  前景比例={r['fg_ratio_mean']:.4%}  全部輪廓={r['contours_all_mean']:.1f}  面積<300={r['blobs_lt300_mean']:.1f}")


if __name__ == "__main__":
    rows = []
    method1(rows)
    method2(rows)
    keys = sorted({k for r in rows for k in r})
    e.RESULT_DIR.mkdir(exist_ok=True)   # 全新環境尚無 results/ 時也能執行
    with open(e.RESULT_DIR / "noise_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n已輸出：", e.RESULT_DIR / "noise_check.csv")
