from transformers import AutoTokenizer, PegasusForConditionalGeneration
from typing import List

def generate_text_summary(text):
    mname = "google/pegasus-xsum"
    model = PegasusForConditionalGeneration.from_pretrained(mname)
    tok = AutoTokenizer.from_pretrained(mname)  

    # don't need tgt_text for inference
    with tok.as_target_tokenizer():
        batch = tok(text, return_tensors='pt')
    gen = model.generate(**batch)  # for forward pass: model(**batch)
    summary: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
    return summary