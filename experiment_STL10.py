def main():
    # Load data
    transform = ToTensor()
    train_set = STL10(root='./datasets', split='train', download=True, transform=transform)
    test_set = STL10(root='./datasets', split='test', download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=64)  
    test_loader = DataLoader(test_set, shuffle=False, batch_size=32)

    # Define model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"{torch.cuda.get_device_name(device)}" if torch.cuda.is_available() else "")
    model = MyViT((3, 96, 96), n_patches=12, n_blocks=6, hidden_d=256, n_heads=8, out_d=10)  
    model = model.to(device)
    N_EPOCHS = 110
    LR = 0.0001

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    train_start_time = time.time()
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{N_EPOCHS} | loss: {train_loss:.2f}")

    epoch_duration = time.time() - train_start_time
    print(f"Training time is: {epoch_duration:.2f} sec")

    # Testing loop
    test_start_time = time.time()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

    test_duration = time.time() - test_start_time
    print(f"Test time is: {test_duration:.2f} sec")

if __name__ == "__main__":
    main()