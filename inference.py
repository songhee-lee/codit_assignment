from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from Korpora import Korpora

corpus = Korpora.load("korean_parallel_koen_news")
article_kr = corpus.train[0].text

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Korean to English
tokenizer.src_lang = "ko_KR"
encoded_ar = tokenizer(article_kr, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(f"KR : {article_kr}")
print(f"EN : {result}")