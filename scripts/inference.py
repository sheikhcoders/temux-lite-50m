"""Simple inference script for Temux-Lite-50M."""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ID = "TheTemuxFamily/Temux-Lite-50M"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, config=config, trust_remote_code=True)
    model.eval()

    prompt = "Temux says: "
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=32)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
