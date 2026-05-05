# El Hareketi Tanıma Projesi

Bu proje, görüntü işleme dersi için Python ile geliştirilmiş bir kamera tabanlı el hareketi tanıma uygulamasıdır.

## Özellikler

- Canlı kamera görüntüsü
- Gerçek zamanlı el tespiti (MediaPipe Hands)
- Temel hareket sınıflandırma:
  - Açık El
  - Yumruk
  - Başparmak Yukarı
  - İşaret
- Profesyonel görünümlü masaüstü arayüz (CustomTkinter)
- Canlı güven skoru ve FPS göstergesi

## Kurulum

1. Python 3.10+ kurulu olduğundan emin olun.
2. Proje klasöründe terminal açın ve şu komutları çalıştırın:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

Uygulama açıldıktan sonra:

1. `Kamerayı Başlat` butonuna basın.
2. Elinizi kadraja getirin.
3. Algılanan hareketi sağ panelde takip edin.

## Aşamalı Geliştirme Planı

### Aşama 1 (Tamamlandı)
- Canlı kamera + temel jest tanıma + modern arayüz.

### Aşama 2
- Daha fazla hareket:
  - Victory (V)
  - OK işareti
  - Avuç içi sola/sağa kaydırma
- Hareket geçmişi paneli ve log kaydı.

### Aşama 3
- Veri toplama modu (her hareket için örnek kaydetme).
- Özel model eğitimi (SVM / MLP).
- Kural tabanlı sınıflandırmadan öğrenme tabanlı sınıflandırmaya geçiş.

### Aşama 4
- Sunum modu:
  - Klavye komutlarına hareket atama (slayt ileri/geri gibi).
- Performans raporu ve hata matrisi.

## Notlar

- İyi aydınlatma ve sade arka plan algılama kalitesini artırır.
- Kameraya erişim izni yoksa uygulama açılmayabilir.
