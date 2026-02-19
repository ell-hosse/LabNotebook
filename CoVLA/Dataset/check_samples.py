import random
from pathlib import Path
import matplotlib.pyplot as plt
import av

root = Path(r"D:\hf")
mp4s = list(root.rglob("*.mp4"))

print("Found mp4 files:", len(mp4s))
assert len(mp4s) > 0, "No mp4s found under D:\\hf"

path = random.choice(mp4s)
print("Random video:", path)

container = av.open(str(path))
frames = []
for i, frame in enumerate(container.decode(video=0)):
    if i % 20 == 0:  # stride
        frames.append(frame.to_rgb().to_ndarray())

container.close()

print(len(frames))

plt.figure(figsize=(12, 6))
for i, img in enumerate(frames):
    ax = plt.subplot(3, 10, i+1)
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.show()
