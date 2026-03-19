# OCL Annotation Web App

React + react-konva tabanlı annotation arayüzü.

## Kurulum

```bash
cd ocl-webapp
npm install
npm run dev
```

Tarayıcıda `http://localhost:5173` açılır.

## Kullanım

1. Herhangi bir fotoğraf yükle (drag & drop veya Dosya Seç)
2. Sol panelden etiket yaz (veya hızlı etiketlere tıkla)
3. Canvas üzerinde nesnenin etrafına daire çiz (mousedown → sürükle → bırak)
4. "SAM'a Gönder" butonuna tıkla

## Backend bağlantısı

`vite.config.ts` içinde `/api` → `http://localhost:8000` proxy tanımlı.  
Backend hazır olduğunda `src/api/client.ts` otomatik bağlanır.

## Dosya yapısı

```
src/
├── App.tsx                  # Ana bileşen
├── components/
│   ├── ImageCanvas.tsx      # react-konva canvas + daire çizimi
│   └── AnnotationSidebar.tsx # Etiket, liste, gönder paneli
├── hooks/
│   └── useAnnotation.ts     # Annotation state yönetimi
├── api/
│   └── client.ts            # Backend API çağrısı
└── types.ts                 # TypeScript tipleri
```
