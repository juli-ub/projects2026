import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

def main():
    base_model_name = "distilgpt2"
    adapter_path = "./saved_adapter"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with AutoAdapterModel support
    model = AutoAdapterModel.from_pretrained(base_model_name)

    print("Loading and activating the Bottleneck Adapter...")
    # Load adapter from the local directory
    adapter_name = model.load_adapter(adapter_path)
    
    # Activate the adapter specifically for inference
    model.set_active_adapters(adapter_name)

    # Define validation prompt
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