import torch
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
from encoder_decoder_utils import load_data, load_model, train

if __name__ == "__main__":

    model_name = sys.argv[1]
    data_name = sys.argv[2]

    model, tokenizer = load_model(model_name)
    if(data_name == "squad"):
        is_squad = True
    else:
        is_squad = False

    train_data, val_data = load_data(is_squad)

    batch_size = 16
    epochs = 3

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, val_losses, trained_model = train(model, train_data, val_data, batch_size, tokenizer, optimizer, epochs, device, is_squad)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("t5_loss_plot_v1.png")
    plt.show()

    torch.save(trained_model, "t5_model_v1.pt")
