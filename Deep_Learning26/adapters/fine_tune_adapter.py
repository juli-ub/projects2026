import torch
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from adapters import AutoAdapterModel, AdapterTrainer

def main():
    model_name = "distilgpt2"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # AutoAdapterModel is the base wrapper from the adapters library
    model = AutoAdapterModel.from_pretrained(model_name)

    # 1. Create a tiny synthetic dataset
    data = {
        "text": [
            "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris.",
            "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of Germany?\n\n### Response:\nThe capital of Germany is Berlin.",
            "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of Italy?\n\n### Response:\nThe capital of Italy is Rome."
        ]
    }
    dataset = Dataset.from_dict(data)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    print("Tokenizing data...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 2. Add a standard Bottleneck Adapter (Pfeiffer architecture)
    # "seq_bn" adds sequential bottleneck layers inside the Transformer blocks.
    adapter_name = "qa_adapter"
    model.add_adapter(adapter_name, config="seq_bn")

    # 3. Freeze all parameters in the model except the newly added adapter
    model.train_adapter(adapter_name)

    # Output parameter details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params} || Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")

    # 4. Set training configurations for CPU
    training_args = TrainingArguments(
        output_dir="./adapter_gpt2_cpu_results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        use_cpu=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Use AdapterTrainer instead of the standard Hugging Face Trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. Run training
    print("\nStarting adapter fine-tuning on CPU...")
    trainer.train()
    print("Fine-tuning completed.")

    # 7. Save only the adapter weights and configuration files
    adapter_save_path = "./saved_adapter"
    model.save_adapter(adapter_save_path, adapter_name)
    print(f"Adapter weights successfully saved to: {adapter_save_path}")

if __name__ == "__main__":
    main()