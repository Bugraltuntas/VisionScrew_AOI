import cv2
import os
import numpy as np
from defect_detector import analyze_and_locate_defects

REF_DIR = "data/reference/"
TEST_DIR = "data/synthetic_samples/"

def main():
    print(f"--- VisionScrew AOI | Mehmet Bugra Altuntas - 225060043 ---")
    
    # 1. Referansları hafızaya yükle
    references = {}
    if not os.path.exists(REF_DIR):
        print(f"Klasor yok: {REF_DIR}")
        return

    for file in os.listdir(REF_DIR):
        if file.endswith(('.webp', '.jpg', '.png')):
            name = file.split('.')[0]
            references[name] = cv2.imread(os.path.join(REF_DIR, file))

    # 2. Test örneklerini tara
    if not os.path.exists(TEST_DIR):
        print(f"Klasor yok: {TEST_DIR}")
        return

    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.webp'))]
    
    for test_file in test_files:
        test_img = cv2.imread(os.path.join(TEST_DIR, test_file))
        
        # Otomatik eşleştirme (Klasördeki isme göre)
        match = None
        for ref_name in references:
            if ref_name in test_file:
                match = ref_name
                break
        
        if match:
            # Analizi patlat
            _, final_view = analyze_and_locate_defects(references[match], test_img)
            
            # PENCERE YÖNETİMİ
            cv2.imshow("Industrial Quality Control Center", final_view)
            
            print(f"Inceleniyor: {test_file} | ESC: Cikis, Diger Tus: Sonraki")
            
            key = cv2.waitKey(0)
            if key == 27: # ESC
                break
        else:
            print(f"Eslesme bulunamadi: {test_file}")

    cv2.destroyAllWindows()
    print("Denetim bitti. Basarilar dilerim aga!")

if __name__ == "__main__":
    main()