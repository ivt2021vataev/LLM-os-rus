from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def main():
    # Создаем пустой BPE токенизатор
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Настраиваем тренера на 15 000 уникальных слов и слогов
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], 
        vocab_size=15000
    )

    print("Обучаем токенизатор на твоем тексте...")
    tokenizer.train(["os_ru_dataset.txt"], trainer)

    # Сохраняем в файл
    tokenizer.save("os_ru_tokenizer.json")
    print("Готово! Файл 'os_ru_tokenizer.json' успешно создан.")

if __name__ == "__main__":
    main()