import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "distilgpt2"
    adapter_path = "./lora/lora_adapted_model"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print("Loading and applying the LoRA adapter...")
    # This merges the small fine-tuned adapter weights back onto the base model at runtime
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Formulate a prompt matching the training format
    prompt = "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nWhat is the capital of Germany?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=15, 
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Model Output ---")
    print(decoded_output)

if __name__ == "__main__":
    main()