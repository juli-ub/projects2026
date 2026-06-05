import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def main():
    # 1. Choose a small model suitable for CPU training
    model_name = "distilgpt2"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 does not have a padding token by default; we set it to the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Create a small dataset for demonstration purposes
    # This keeps the training time short on a CPU
    data = {
        "text": [
            "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris.",
            "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of Germany?\n\n### Response:\nThe capital of Germany is Berlin.",
            "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of Italy?\n\n### Response:\nThe capital of Italy is Rome."
        ]
    }
    dataset = Dataset.from_dict(data)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    print("Tokenizing data...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 3. Configure LoRA
    # In GPT-2 models, the multi-head attention projection layers are named "c_attn"
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                  # Rank of the update matrices
        lora_alpha=32,        # Scaling factor
        lora_dropout=0.1,     # Dropout probability for LoRA layers
        target_modules=["c_attn"] 
    )

    # 4. Wrap the model with PEFT
    peft_model = get_peft_model(model, peft_config)
    print("\nModel Parameter Summary:")
    peft_model.print_trainable_parameters()

    # 5. Set up training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir="./lora_gpt2_cpu_results",
        num_train_epochs=5,             # Low epoch count for a fast demonstration
        per_device_train_batch_size=1,  # Keep batch size small for memory efficiency
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",             # Skip checkpoint saving to save disk space during training
        use_cpu=True,                   # Force training to run on CPU
    )

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. Start training
    print("\nStarting fine-tuning on CPU...")
    trainer.train()
    print("Fine-tuning completed.")

    # 8. Save the LoRA adapter
    # This only saves the lightweight adapter weights (~2MB), not the entire base model.
    adapter_path = "./lora/lora_adapted_model"
    os.makedirs(adapter_path, exist_ok=True)
    peft_model.save_pretrained(adapter_path)
    print(f"LoRA adapter weights saved to: {adapter_path}")

if __name__ == "__main__":
    main()