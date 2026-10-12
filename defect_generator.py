import cv2
import numpy as np
import os
import random

class DefectGenerator:
    def __init__(self, output_dir="data/synthetic_samples/"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def add_rust(self, img):
        """Vidanın üzerine rastgele pas (oksitlenme) lekeleri ekler."""
        rust_color = [random.randint(20, 50), random.randint(50, 100), random.randint(100, 180)] # Kahverengi tonları
        mask = np.zeros(img.shape[:2], np.uint8)
        for _ in range(5): # 5 tane pas lekesi
            center = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
            axes = (random.randint(10, 40), random.randint(5, 20))
            cv2.ellipse(mask, center, axes, random.randint(0, 360), 0, 360, 255, -1)
        
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        img_rust = img.copy()
        for c in range(3):
            img_rust[:, :, c] = np.where(mask > 100, (img[:, :, c] * 0.6 + rust_color[c] * 0.4).astype(np.uint8), img[:, :, c])
        return img_rust

    def add_burr(self, img):
        """Vidanın kenarlarına üretim hatası (çapak) ekler."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        burr_img = img.copy()
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            for _ in range(3): # 3 tane çapak noktası
                idx = random.randint(0, len(cnt)-1)
                pt = cnt[idx][0]
                cv2.circle(burr_img, tuple(pt), random.randint(5, 15), (180, 180, 180), -1)
        return burr_img

    def add_broken_tip(self, img):
        """Vidanın ucunu matematiksel olarak 'kırar' (maskeler)."""
        h, w = img.shape[:2]
        broken_img = img.copy()
        # Vidanın ucuna (genellikle sağ veya sol uç) beyaz maske atar
        cv2.rectangle(broken_img, (w-random.randint(30, 60), 0), (w, h), (255, 255, 255), -1)
        return broken_img

    def process_all(self, ref_folder):
        for file in os.listdir(ref_folder):
            if file.endswith(('.webp', '.jpg', '.png')):
                path = os.path.join(ref_folder, file)
                img = cv2.imread(path)
                name = file.split('.')[0]
                
                # Her vida tipi için 5 farklı hata üret
                for i in range(1, 6):
                    defect_type = random.choice(['rust', 'burr', 'broken'])
                    if defect_type == 'rust':
                        result = self.add_rust(img)
                    elif defect_type == 'burr':
                        result = self.add_burr(img)
                    else:
                        result = self.add_broken_tip(img)
                    
                    cv2.imwrite(f"{self.output_dir}{name}_defect_{i}.jpg", result)
                    print(f"Başarılı: {name} için {defect_type} hatası üretildi.")

if __name__ == "__main__":
    gen = DefectGenerator()
    gen.process_all("data/reference/")