import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from tqdm import tqdm
from pathlib import Path

from Fusion.residual_adapter import ResidualAdapterFusionTextMain
from Temporal_models.LNN import LNN
from Preprocessing.traj_2_num import traj_jsonl_2_num


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            total += criterion(pred, y).item()

    return total / len(loader)


def evaluate_metrics(model, loader, device):
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            dist = torch.norm(pred - y, dim=-1)

            batch_size = y.size(0)

            ade = dist.mean(dim=1)
            fde = dist[:, -1]

            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            total_samples += batch_size

    return total_ade / total_samples, total_fde / total_samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_model = ResidualAdapterFusionTextMain(embed_dim=512, adapter_dim=128).to(device)

    lnn_model = LNN(
        input_dim=512,
        hidden_dim=256,
        traj_points=10,
        coord_dim=3,
        dropout=0.1,
    ).to(device)

    with torch.no_grad():
        fused_embeddings = fusion_model.creat_fused_embeddings().float()

    fused_embeddings = fused_embeddings.detach().cpu()

    traj = traj_jsonl_2_num()
    traj = torch.from_numpy(traj).float()

    for i in range(5):
        print(i)
        print("embedding norm:", fused_embeddings[i].norm().item())
        print("traj norm:", traj[i].norm().item())

    dataset = TensorDataset(fused_embeddings[: 2248], traj[: 2248])

    split_path = Path(r"D:\hf\CoVLA-metadata\data_splits.pt")

    if split_path.exists():
        splits = torch.load(split_path)
        train_set = Subset(dataset, splits["train"])
        val_set = Subset(dataset, splits["val"])
        test_set = Subset(dataset, splits["test"])
    else:
        n = len(dataset)
        train_n = int(0.8 * n)
        val_n = int(0.1 * n)
        test_n = n - train_n - val_n

        gen = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = random_split(
            dataset,
            [train_n, val_n, test_n],
            generator=gen
        )

        torch.save(
            {
                "train": train_set.indices,
                "val": val_set.indices,
                "test": test_set.indices,
            },
            split_path,
        )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lnn_model.parameters(), lr=1e-3)

    save_path = Path(r"D:\hf\CoVLA-metadata\best_model.pth")

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    epochs = 100

    for epoch in range(epochs):
        lnn_model.train()
        total = 0.0

        loop = tqdm(train_loader, leave=True)
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = lnn_model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total += loss.item()
            loop.set_description(f"Epoch {epoch + 1}/{epochs}")
            loop.set_postfix(loss=loss.item())

        train_loss = total / len(train_loader)
        val_loss = evaluate_loss(lnn_model, val_loader, criterion, device)
        val_ade, val_fde = evaluate_metrics(lnn_model, val_loader, device)

        print(
            f"Epoch {epoch + 1} | "
            f"train_mse {train_loss:.6f} | "
            f"val_mse {val_loss:.6f} | "
            f"val_ADE {val_ade:.6f} | "
            f"val_FDE {val_fde:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(lnn_model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    lnn_model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss = evaluate_loss(lnn_model, test_loader, criterion, device)
    test_ade, test_fde = evaluate_metrics(lnn_model, test_loader, device)

    print(f"test_mse {test_loss:.6f}")
    print(f"test_ADE {test_ade:.6f}")
    print(f"test_FDE {test_fde:.6f}")


if __name__ == "__main__":
    main()