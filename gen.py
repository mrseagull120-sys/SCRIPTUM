from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "Once upon a time in a forgotten forest,"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_length=300,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))