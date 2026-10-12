import cv2
import numpy as np

def analyze_and_locate_defects(ref_img, test_img):
    # 1. STANDART BOYUTLANDIRMA
    h, w = ref_img.shape[:2]
    test_resized = cv2.resize(test_img, (w, h))
    
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)

    # 2. FARK ANALİZİ VE İŞLEME
    diff = cv2.absdiff(ref_gray, test_gray)
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((15, 15), np.uint8)
    diff_dilated = cv2.dilate(diff_thresh, kernel, iterations=2)
    
    # 3. KONTUR VE DAİRE ÇİZİMİ
    contours, _ = cv2.findContours(diff_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = test_resized.copy()
    defect_count = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            defect_count += 1
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            cv2.circle(result_img, center, int(radius) + 15, (0, 0, 255), 3)
            cv2.putText(result_img, "HATA!", (int(x)-20, int(y)-int(radius)-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4. MSE HESAPLAMA
    mse_value = np.sum((ref_gray.astype("float") - test_gray.astype("float")) ** 2)
    mse_value /= float(ref_gray.shape[0] * ref_gray.shape[1])

    # 5. AKILLI BOYUTLANDIRMA (EKRANA SIĞDIRMA)
    # Genişliği 600'e çekiyoruz ki dikeyde çok uzamasın
    target_w = 650 
    
    ref_h = int(ref_img.shape[0] * (target_w / ref_img.shape[1]))
    ref_disp = cv2.resize(ref_img, (target_w, ref_h))
    
    test_h = int(result_img.shape[0] * (target_w / result_img.shape[1]))
    test_disp = cv2.resize(result_img, (target_w, test_h))

    # Etiketler
    status = "OK" if defect_count == 0 else f"REJECTED ({defect_count} Hata)"
    color = (0, 255, 0) if defect_count == 0 else (0, 0, 255)
    
    cv2.putText(ref_disp, "REFERANS (GOLDEN)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(test_disp, f"SONUC: {status} | MSE: {mse_value:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ALT ALTA BİRLEŞTİR
    sep = np.zeros((10, target_w, 3), dtype=np.uint8)
    combined = np.vstack((ref_disp, sep, test_disp))

    # SON KONTROL: Eğer hala çok uzunsa (800 pikselden fazlaysa) komple küçült
    if combined.shape[0] > 850:
        final_h = 850
        final_w = int(combined.shape[1] * (final_h / combined.shape[0]))
        combined = cv2.resize(combined, (final_w, final_h))

    return defect_count == 0, combined