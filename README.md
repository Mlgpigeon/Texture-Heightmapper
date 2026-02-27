# ⛰️ Heightmap Studio

Herramienta para generar heightmaps a partir de texturas de Pokémon (o cualquier textura).
Detecta regiones de color, permite asignar alturas individuales, y exporta un grayscale PNG
listo para usar como displacement map en Blender.

## Setup rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar
python server.py

# 3. Abrir en el navegador
# http://localhost:8000
```

## Uso

1. **Sube una textura** (drag & drop o click)
2. **Elige un algoritmo** de detección:
   - **Componentes Conectados**: Regiones por color + proximidad. Mismo color en zonas separadas = regiones distintas.
   - **Clustering por Color**: Agrupa por color sin importar posición.
   - **Superpíxeles (SLIC)**: Segmentación perceptual, mejor para gradientes.
3. **Ajusta parámetros** (tolerancia, tamaño mínimo de región, etc.)
4. **Click "Detectar Regiones"**
5. **Asigna alturas** con los sliders (0=negro/hundido, 255=blanco/elevado)
6. **Descarga** el heightmap PNG

### Features extra
- **Hover** sobre una región → se resalta en la imagen
- **Fusionar** regiones que el algoritmo separó incorrectamente
- **Presets** rápidos: Claro↑, Oscuro↑, Por área, Plano

## Arquitectura

```
heightmap-studio/
├── server.py                  # FastAPI — API REST + sirve frontend
├── processors/
│   ├── base.py                # BaseProcessor — interfaz para todos los algoritmos
│   ├── connected_components.py # Flood fill con tolerancia de color
│   ├── color_cluster.py       # Clustering por color puro
│   └── superpixels.py         # SLIC superpixels (scikit-image)
├── static/
│   └── index.html             # Frontend standalone
└── requirements.txt
```

## Añadir un nuevo procesador

1. Crea `processors/mi_procesador.py`:

```python
from .base import BaseProcessor, Region, DetectionResult
import numpy as np

class MiProcesador(BaseProcessor):
    name = "Mi Algoritmo"
    description = "Descripción para el UI"

    def get_params(self) -> list[dict]:
        return [
            {"key": "param1", "label": "Parámetro", "type": "slider",
             "min": 0, "max": 100, "default": 50, "hint": "..."},
        ]

    def detect(self, image: np.ndarray, params: dict) -> DetectionResult:
        # image es (H, W, 4) RGBA uint8
        # Devuelve DetectionResult con label_map y lista de Region
        ...
```

2. Regístralo en `processors/__init__.py`:
```python
from .mi_procesador import MiProcesador
_register(MiProcesador)
```

3. Reinicia el server → aparece automáticamente en el dropdown del UI.

### Ideas para procesadores futuros
- **SAM (Segment Anything)**: Segmentación por IA con prompts
- **Edge-based**: Detectar bordes con Canny/Sobel y usarlos como límites
- **Watershed**: Segmentación basada en topología de la imagen
- **Manual paint**: Pintar regiones directamente en el canvas
- **From normal map**: Derivar alturas desde un normal map existente

## API REST

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/processors` | GET | Lista procesadores disponibles |
| `/api/upload` | POST | Sube imagen, devuelve session_id |
| `/api/detect/{sid}` | POST | Ejecuta detección |
| `/api/update_heights/{sid}` | POST | Actualiza alturas de regiones |
| `/api/merge/{sid}` | POST | Fusiona dos regiones |
| `/api/render/{sid}/{mode}` | GET | Renderiza vista como PNG |
| `/api/download/{sid}` | GET | Descarga heightmap final |
