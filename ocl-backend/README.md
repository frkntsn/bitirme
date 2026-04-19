# OCL Backend

FastAPI + SAM + DINOv2 gömü (torch.hub) ve isteğe bağlı CLIP yedeği; NCM + replay buffer ile online güncelleme.

## Kurulum

```bash
cd ocl-backend

# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Paketler
pip install -r requirements.txt

# 3. SAM ağırlığını indir (~375 MB)
mkdir -p weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P weights/

# 4. .env oluştur
cp .env.example .env
# DEVICE=mps  → MacBook M serisi
# DEVICE=cpu  → GPU yoksa
```

## Çalıştır

```bash
uvicorn main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## Endpoint'ler

| Method | Path | Açıklama |
|--------|------|---------|
| GET | /health | SAM + VLM yüklü mü? Buffer boyutu? |
| POST | /reset-memory | NCM + buffer + `data/online_memory.pkl` sıfırla (yalnızca siz unutturunca) |
| POST | /segment | Segmentasyon + tahmin + online update |

### POST /segment

**Request:**
```json
{
  "image": "<base64>",
  "annotations": [
    { "x": 312, "y": 245, "radius": 60, "label": "tümör" }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "label": "tümör",
      "predicted_class": "tümör",
      "confidence": 0.87,
      "mask_area": 4821
    }
  ],
  "model_updated": true
}
```

## MacBook M serisi notu

`.env` içinde `DEVICE=mps` yaz; DINOv2 MPS üzerinde çalışır.  
SAM (`segment_anything`) tam görsel maskesinde MPS + float64 hatası verebildiği için varsayılan olarak **SAM ayrı cihazda CPU** çalışır (`SAM_DEVICE` boşken `DEVICE=mps` → `SAM_DEVICE=cpu`). İstersen `.env` ile `SAM_DEVICE=mps` deneyebilirsin; hata alırsan `cpu` bırak.

Tam görsel tarama için `opencv-python-headless` kurulu olmalı (SAM `SamAutomaticMaskGenerator`).

## Kalıcı sınıf hafızası (.env, isteğe bağlı)

Öğrenilen NCM + replay buffer varsayılan olarak `ocl-backend/data/online_memory.pkl` dosyasına yazılır; sunucu yeniden başlasa da korunur. Unutmak için yalnızca `POST /reset-memory` kullanın.

- `ONLINE_MEMORY_ENABLED=0` — diske yazma/okuma kapalı (her restart’ta boş).
- `ONLINE_MEMORY_PATH` — farklı dosya yolu.

## Tespit ayarları (.env, isteğe bağlı)

- `DETECTION_THRESHOLD` — varsayılan 0.65; çok gürültü varsa yükselt (örn. 0.72).
- `DETECTION_TOP_K` — en fazla kaç kutu dönsün (varsayılan 30).
- `DETECTION_NMS_IOU` — üst üste binen kutuları birleştirme eşiği (varsayılan 0.40).
- `SCAN_FALLBACK=sliding|grid` — SAM auto yokken: kaydırmalı pencere veya kaba grid.
- `SAM_AUTO_MIN_MASK_AREA` — küçük hücreler için düşürülebilir (örn. 80).

## Dosya yapısı

```
ocl-backend/
├── main.py           # FastAPI app + endpoint'ler
├── segmentation.py   # SAM wrapper
├── vlm_model.py      # DINOv2 + MIR buffer + NCM classifier
├── schemas.py        # Pydantic modeller
├── config.py         # Tüm ayarlar (.env'den okur)
├── requirements.txt
├── .env.example
└── weights/          # SAM ağırlık dosyası buraya
    └── sam_vit_b_01ec64.pth
```
