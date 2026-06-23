import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 15000,  
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class OsetianDataset(Dataset):
    def __init__(self, tokens, context_length, stride=256):
        self.tokens = tokens
        self.context_length = context_length
        self.stride = stride

    def __len__(self):
        return (len(self.tokens) - self.context_length) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        return (
            torch.tensor(self.tokens[start_idx : start_idx + self.context_length]),
            torch.tensor(self.tokens[start_idx + 1 : start_idx + self.context_length + 1])
        )

def train():
    with open('os_ru_dataset.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()

    print("Кодируем текст новым осетинским токенизатором...")
    tokenizer = Tokenizer.from_file("os_ru_tokenizer.json")
    tokens = tokenizer.encode(text_data).ids 
    print(f"Всего токенов в датасете: {len(tokens)}")

    dataset = OsetianDataset(tokens, context_length=256, stride=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Батчей в одной эпохе: {len(dataloader)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучение запущено на: {device.upper()}")

    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    model.train()
    print("Начинаем обучение на эпох...")
    
    for epoch in range(30):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if i % 100 == 0:
                print(f"Эпоха {epoch} | Шаг {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model_final.pth")
    print("Обучение полностью завершено! Веса сохранены в 'model_final.pth'")


if __name__ == "__main__":
    train()