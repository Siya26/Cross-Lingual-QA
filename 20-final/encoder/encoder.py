import torch
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertForQuestionAnswering
from encoder_utils import train, load_data, load_model
 
if __name__ == "__main__":
    model_name = sys.argv[1]
    data_name = sys.argv[2]

    model, tokenizer = load_model(model_name)
    train_data, val_data = load_data(data_name)

    batch_size = 128
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, val_losses, trained_model = train(model, model_name, train_data, val_data, batch_size, tokenizer, optimizer, epochs, device)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_name}_loss_plot.png")
    plt.show()



 