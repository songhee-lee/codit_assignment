import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from Korpora import Korpora
from tqdm import tqdm

class TranslationDataset(Dataset) :
    def __init__(self, data, tokenizer, max_length=512) :
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) :
        return len(self.data.text)

    def __getitem__(self, idx) :
        source = self.data.text[idx]
        target = self.data.pair[idx]

        source_encodings = self.tokenizer(source, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        target_encodings = self.tokenizer(target, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids' : source_encodings['input_ids'].squeeze(0),
            'attention_mask' : source_encodings['attention_mask'].squeeze(0),
            'labels' : target_encodings['input_ids'].squeeze(0)
        }


def finetuning_mbart() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    # 데이터셋 로드
    dataset_name = "korean_parallel_koen_news"
    batch_size = 8

    corpus = Korpora.load(dataset_name)
    train_dataset = TranslationDataset(corpus.train[:10], tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 학습
    learning_rate=5e-5
    epochs=1

    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs) :
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in progress_bar :
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1} Step {step +1} Loss {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    
    # 모델 저장
    model_path = "./"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__" :
    finetuning_mbart()