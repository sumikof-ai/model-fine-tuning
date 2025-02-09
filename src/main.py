from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# 1. アセンブリコードのデータセット作成例（ここでは簡単なサンプル）
data = {
    "code": [
        "MOV AX, BX\nADD AX, 1\nRET", 
        "PUSH EBP\nMOV EBP, ESP\nSUB ESP, 16\nPOP EBP\nRET"
    ],
    "label": [0, 1]  # タスクに応じたラベル付け
}
dataset = Dataset.from_dict(data)
# 実際はもっと大規模なデータセットが必要です

# 2. トークナイザーの準備（必要に応じて新たな語彙学習を実施）
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
def tokenize_function(example):
    return tokenizer(example["code"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. モデルの読み込み（例としてシーケンス分類タスク用）
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)

# 4. ファインチューニングの設定
training_args = TrainingArguments(
    output_dir="./assembly_code_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # 評価データは分割したデータを利用するのが望ましい
    tokenizer=tokenizer,
)

# 5. ファインチューニングの実行
trainer.train()

# 6. モデル保存
model.save_pretrained("./fine-tuned-assembly-code-model")
tokenizer.save_pretrained("./fine-tuned-assembly-code-model")