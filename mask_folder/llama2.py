import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model name
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False  # greedy decoding, deterministic
)


# Define PII types
valid_PII_classes = ["GIVENNAME", "SURNAME", "STREET", "DATEOFBIRTH", "CITY"]

# --------
# MODE: choose "detect" or "mask"
# --------
mode = "detect"
def llama_pii_task(text, mode="detect"):
    if mode == "detect":
        prompt = f"""You are a realistic token filler. Whenever you receive a sentence as an input containing tokens from tokens_list, you will only replace each token with a plausible, realistic value that fits naturally in the context.

Guidelines:
- Replace tokens with completely **new, unique human-like values**, not from a predefined list. (e.g., [SURNAME] → Johnson, [CITY] → Chicago, [EMAIL] → mark.smith@example.com).
- Ensure contextual consistency (e.g., if [SURNAME] appears multiple times, use the same surname).
- Vary replacements based on context (e.g., [EMAIL] should match a professional or personal setting).
- Keep the output fluent and natural, as if written by a human.
- **Do not modify any words** in the input sentences other than replacing the tokens.
- You must be consistent while filling tokens. For example if a name is 'ali yasar' email should be similar to 'ali@gmail.com', 'a.yasar@email.com' or 'ali.yasar@hotmail.com'.
IMPORTANT NOTE: DO NOT PRINT any reasoning other than the modified text itself.

Here is the token list:
tokens_list = [
    "[ACCOUNTNUM]", "[BUILDINGNUM]", "[CITY]", "[CREDITCARDNUMBER]", "[DATEOFBIRTH]",
    "[DRIVERLICENSENUM]", "[EMAIL]", "[GIVENNAME]", "[IDCARDNUM]", "[PASSWORD]",
    "[SOCIALNUM]", "[STREET]", "[SURNAME]", "[TAXNUM]", "[TELEPHONENUM]", "[USERNAME]", "[ZIPCODE]"
]

Text: {text}
"""
    else:  # mask mode
        prompt = f"""You are a PII masker.
Find PII of these categories in the text: GIVENNAME, SURNAME, DATEOFBIRTH, CITY.
Replace each with its label in square brackets.
Example:
Input: John Smith was born in Paris on 12 March 1980.
Output: [GIVENNAME] [SURNAME] was born in [CITY] on [DATEOFBIRTH].

Text: {text}
"""
    out = pipe(prompt)[0]["generated_text"]
    response = out[len(prompt):].strip()
    return response

# ---- Load Data ----
file_path = "input_10.xlsx"
df = pd.read_excel(file_path)

# ---- Apply to DataFrame ----
df["llama_output"] = df["original_text"].apply(lambda x: llama_pii_task(x, mode))

# Save results
df.to_excel("llama2_output.xlsx", index=False)
print("Done. Check 'llama_output' column in llama2_output.xlsx.")
