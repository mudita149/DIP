from PIL import Image
import heapq
from collections import Counter
import matplotlib.pyplot as plt

# Huffman Node
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(freq_dict):
    heap = [Node(sym, freq) for sym, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    codes = {}
    def generate_codes(node, code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code
            return
        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")
    generate_codes(heap[0])
    return codes

def encode(pixels, codes):
    return "".join(codes[p] for p in pixels)

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

# Load grayscale image (replace with your image path)
img = Image.open(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg").convert("L")
pixels = list(img.getdata())
freq = Counter(pixels)

# Huffman
codes = huffman_coding(freq)
encoded = encode(pixels, codes)
decoded = decode(encoded, codes, len(pixels))
decoded_img = Image.new("L", img.size)
decoded_img.putdata(decoded)

# Stats
print("Original size (bits):", len(pixels) * 8)
print("Compressed size (bits):", len(encoded))
print("Compression ratio:", round((len(pixels) * 8) / len(encoded), 3))

# Save images
# img.save("huffman_original_image.jpeg")
decoded_img.save("huffman_decoded.jpeg")
print("Image saved as 'huffman_decoded.jpeg'")

# Display images
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original")
axs[1].imshow(decoded_img, cmap='gray')
axs[1].set_title("Huffman Decoded")
for ax in axs:
    ax.axis("off")
plt.show()
