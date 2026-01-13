import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

r3m = load_r3m("resnet50")
r3m.eval()
r3m.to(device)

transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

image = np.random.randint(0, 255, (500, 500, 3))
preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
with torch.no_grad():
  embedding = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]
print(embedding.shape) # [1, 2048]

plt.imshow(image.astype(np.uint8))
plt.title("Input image")
plt.axis("off")

emb = embedding.squeeze().cpu().numpy()


# Plot the embedding as a feature activation profile
plt.figure(figsize=(10, 3))
plt.plot(emb)
plt.title("R3M embedding (2048D)")
plt.xlabel("Feature index")
plt.ylabel("Activation")
plt.tight_layout()

# Plot embedding as a heatmap
plt.figure(figsize=(12, 2))
plt.imshow(emb[None, :], aspect="auto", cmap="viridis")
plt.colorbar(label="Activation")
plt.yticks([])
plt.xlabel("Feature index")
plt.title("R3M embedding heatmap")

plt.show()
