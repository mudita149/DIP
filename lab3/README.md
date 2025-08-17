# Lab 3 — Lossless Image Compression (Huffman & Shannon–Fano)

This lab implements and compares two classic **lossless** compression methods on a grayscale image:

- **Huffman coding** (`huffman.py`)
- **Shannon–Fano coding** (`shannonfano.py`)

Both scripts:
- Read `earth.jpeg` in **grayscale**
- Build a codebook from pixel **frequencies (0–255)**
- **Encode** all pixels to a bitstring
- **Decode** back to pixels and reconstruct the image
- Print basic compression stats
- Display **Original vs Decoded** images

> Decoded images are saved as:
> - `huffman_decoded.jpeg`
> - `shannon_fano_decoded.png`

---

## Folder Structure
```
Lab_3/
├─ huffman.py
├─ shannonfano.py
├─ earth.jpeg
├─ huffman_decoded.jpeg #o/p of huffman
└─ shannon_fano_decoded.png #o/p of shannonfano

---
