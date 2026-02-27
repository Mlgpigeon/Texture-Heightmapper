export function regionColor(i) {
  const hue = (i * 137.508) % 360
  const s = 0.8 + (i % 3) * 0.1
  const l = 0.5 + (i % 2) * 0.15
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs((hue / 60) % 2 - 1))
  const m = l - c / 2
  let r, g, b
  if (hue < 60)       { r = c; g = x; b = 0 }
  else if (hue < 120) { r = x; g = c; b = 0 }
  else if (hue < 180) { r = 0; g = c; b = x }
  else if (hue < 240) { r = 0; g = x; b = c }
  else if (hue < 300) { r = x; g = 0; b = c }
  else                { r = c; g = 0; b = x }
  return [((r + m) * 255) | 0, ((g + m) * 255) | 0, ((b + m) * 255) | 0]
}

export function typedArrayToBase64(arr) {
  const bytes = new Uint8Array(arr.buffer || arr)
  let binary = ''
  const chunk = 8192
  for (let i = 0; i < bytes.byteLength; i += chunk)
    binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + chunk, bytes.byteLength)))
  return btoa(binary)
}

export function base64ToInt32Array(b64) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Int32Array(bytes.buffer)
}

export function imgRGBAtoDataURL(rgba, width, height) {
  const c = document.createElement('canvas')
  c.width = width; c.height = height
  c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0)
  return c.toDataURL('image/png')
}

export function debounce(fn, ms) {
  let t = null
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms) }
}
