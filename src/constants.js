export const PROCESSORS = {
  connected: {
    name: 'Componentes Conectados',
    desc: 'Detecta regiones por color + proximidad espacial. Mismo color en zonas separadas = regiones distintas.',
    params: [
      { key: 'tolerance', label: 'Tolerancia de color', type: 'range', min: 5, max: 80, step: 1, default: 30 },
      { key: 'min_region_px', label: 'Región mínima (px)', type: 'range', min: 1, max: 2000, step: 1, default: 50 },
      { key: 'connectivity', label: 'Conectividad', type: 'select', options: [{v:4,l:'4-vecinos'},{v:8,l:'8-vecinos'}], default: 4 },
    ],
  },
  color_cluster: {
    name: 'Clustering por Color',
    desc: 'Agrupa píxeles solo por color (ignora posición). Todos los píxeles del mismo color = misma región.',
    params: [
      { key: 'tolerance', label: 'Tolerancia de color', type: 'range', min: 5, max: 80, step: 1, default: 35 },
      { key: 'min_region_px', label: 'Región mínima (px)', type: 'range', min: 1, max: 2000, step: 1, default: 50 },
      { key: 'max_samples', label: 'Muestras', type: 'range', min: 5000, max: 100000, step: 5000, default: 30000 },
    ],
  },
}
