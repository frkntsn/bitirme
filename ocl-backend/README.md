# OCL Backend

FastAPI + SAM + CLIP tabanlı segmentasyon ve online continual learning backend.

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

`.env` içinde `DEVICE=mps` yaz. PyTorch MPS backend ile GPU hızlanması alırsın.  
SAM ve CLIP ikisi de MPS destekler.

## Dosya yapısı

```
ocl-backend/
├── main.py           # FastAPI app + endpoint'ler
├── segmentation.py   # SAM wrapper
├── vlm_model.py      # CLIP + MIR buffer + NCM classifier
├── schemas.py        # Pydantic modeller
├── config.py         # Tüm ayarlar (.env'den okur)
├── requirements.txt
├── .env.example
└── weights/          # SAM ağırlık dosyası buraya
    └── sam_vit_b_01ec64.pth
```
