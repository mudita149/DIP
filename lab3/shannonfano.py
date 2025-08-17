from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# Shannon–Fano Coding
def shannon_fano(freq_dict):
    symbols = list(freq_dict.items())
    symbols.sort(key=lambda x: x[1], reverse=True)
    codes = {sym: "" for sym, _ in symbols}

    def divide(start, end):
        if start >= end:
            return
        total = sum(freq for _, freq in symbols[start:end+1])
        acc = 0
        split = start
        for i in range(start, end+1):
            if acc + symbols[i][1] <= total / 2:
                acc += symbols[i][1]
                split = i
            else:
                break
        for i in range(start, split+1):
            codes[symbols[i][0]] += "0"
        for i in range(split+1, end+1):
            codes[symbols[i][0]] += "1"
        divide(start, split)
        divide(split+1, end)

    divide(0, len(symbols) - 1)
    return codes

# Encode image pixels
def encode(pixels, codes):
    return "".join(codes[p] for p in pixels)

# Decode bitstring
def decode(bitstring, codes, size):
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_pixels = []
    code = ""
    for bit in bitstring:
        code += bit
        if code in reverse_codes:
            decoded_pixels.append(reverse_codes[code])
            code = ""
    return decoded_pixels[:size]

# === Main Program ===
# Load grayscale image
img = Image.open(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg").convert("L")  # Change to your image path
pixels = list(img.getdata())
freq = Counter(pixels)

# Shannon–Fano
codes = shannon_fano(freq)
encoded = encode(pixels, codes)
decoded = decode(encoded, codes, len(pixels))

# Reconstruct image from decoded pixels
decoded_img = Image.new("L", img.size)
decoded_img.putdata(decoded)

# Save images
# img.save("original_image.png")
decoded_img.save("shannon_fano_decoded.png")

# Stats
original_bits = len(pixels) * 8
compressed_bits = len(encoded)
ratio = round(compressed_bits / original_bits, 3)

print("Original size (bits):", original_bits)
print("Compressed size (bits):", compressed_bits)
print("Compression ratio:", ratio)
print("Image saved as 'shannon_fano_decoded.png'")

# Display images
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original Image")
axs[1].imshow(decoded_img, cmap='gray')
axs[1].set_title("Shannon–Fano Decoded")
for ax in axs:
    ax.axis("off")
plt.show()
