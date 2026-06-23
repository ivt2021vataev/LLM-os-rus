import torch
from tokenizers import Tokenizer
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск генерации на: {device.upper()}")

    model = GPTModel(GPT_CONFIG_124M)
    
    try:
        model.load_state_dict(torch.load("model_final.pth", map_location=device))
        print("Успешно загружены новые веса модели.")
    except FileNotFoundError:
        print("[ОШИБКА] Файл весов 'model_final.pth' не найден. Сначала запусти train.py!")
        return

    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_file("os_ru_tokenizer.json")
    
    input_text = "Осетинское слово «салам»"
    input_ids = tokenizer.encode(input_text).ids
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    print("Генерация...")
    output_ids = model.generate(
        idx=input_tensor, 
        max_new_tokens=20,  
        context_size=256    
    )
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    print("\n" + "="*40)
    print("РЕЗУЛЬТАТ С НОВЫМ ТОКЕНИЗАТОРОМ:")
    print("="*40)
    print(generated_text)
    print("="*40)


if __name__ == "__main__":
    main()