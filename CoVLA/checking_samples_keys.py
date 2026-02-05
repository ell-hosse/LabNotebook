from datasets import load_dataset, Video

ds = load_dataset(
    "turing-motors/CoVLA-Dataset",
    split="train",
    streaming=True,
)

# Disable decoding
ds = ds.cast_column("video", Video(decode=False))

# grab a single row
sample = next(iter(ds))

print("TYPE(sample):", type(sample))
print("\nKEYS:", list(sample.keys()))

print("\n--- preview text-ish fields ---")
for k, v in sample.items():
    if isinstance(v, str):
        print(f"{k}: {v[:250]}")
    else:
        print(f"{k}: {type(v)} -> {v if k=='video' else ''}")
