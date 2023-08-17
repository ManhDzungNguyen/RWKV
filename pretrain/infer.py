from transformers import AutoTokenizer, RwkvForCausalLM
from data_handler import text_normalize

model_path = "RWKV/pretrain/checkpoint/checkpoint-640"
model = RwkvForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

while True:
    context = input("prompt: ")
    if context == "exit":
        break

    inputs = tokenizer(text_normalize(context), return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        no_repeat_ngram_size=5,
        do_sample=True,
        top_k=100,
        top_p=0.95,
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
