import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH  = r"C:\Users\Bochra\CoachMe\model\phi3-fintess-app\checkpoint-609"

SYSTEM_PROMPT = """You are an expert AI fitness and nutrition coach. 
Give structured responses using headers and bullet points.
Be specific with numbers (sets, reps, calories, macros)."""

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

print("Loading base model on CPU... (downloads ~2.3GB first time, be patient)")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,   # CPU needs float32
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading your LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print("✅ Ready! Ask your fitness coach anything.\n")

def ask(question, domain="FITNESS"):
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n[{domain}] {question}<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")  # no .to("cuda")
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,        # lower = faster on CPU
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.15,
            eos_token_id=[tokenizer.eos_token_id, end_token_id],
            pad_token_id=tokenizer.eos_token_id,
        )

    full  = tokenizer.decode(out[0], skip_special_tokens=False)
    reply = full.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
    return reply

# Chat loop
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break
    domain = "NUTRITION" if any(w in question.lower() for w in ["calorie","protein","eat","diet","macro","food"]) else "FITNESS"
    print(f"\nCoach [{domain}]: (generating, please wait...)")
    print(ask(question, domain))
    print()