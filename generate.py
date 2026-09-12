import json,torch
from tokenizer import SentencePieceTokenizer
from model import MiniLLM
try: from torch.ao.quantization import quantize_dynamic
except Exception: quantize_dynamic=None
cfg=json.load(open("config.json",encoding="utf-8")); tok=SentencePieceTokenizer("data/tokenizer.model")
ck=torch.load("data/model.pt",map_location="cpu",weights_only=False); cfg["vocab_size"]=tok.vocab_size
for k in ("n_embd","n_head","n_layer","block_size","dropout"):
    if k in ck.get("model_config",{}): cfg[k]=ck["model_config"][k]
model=MiniLLM(cfg); model.load_state_dict(ck["model"]); model.eval()
if cfg.get("inference",{}).get("quantized",False) and quantize_dynamic: model=quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8); model.eval()
def generate(prompt):
    ids=tok.encode(prompt,add_bos=True,add_eos=False)
    x=torch.tensor([ids],dtype=torch.long)
    inf=cfg.get("inference",{})
    out=model.generate(x,max_new_tokens=int(inf.get("max_new_tokens",120)),temperature=float(inf.get("temperature",.65)),top_k=int(inf.get("top_k",30)),repetition_penalty=float(inf.get("repetition_penalty",1.15)),no_repeat_ngram_size=int(inf.get("no_repeat_ngram_size",3)),eos_token_id=tok.eos_id)
    text=tok.decode(out[0].tolist()[len(ids):]).strip()
    import re, unicodedata
    for marker in ('\nユーザー:', '\nAI:', '\n現在:', '\n返信:', '\n回复:', '<|end_document|>', '<|document|>'):
        if marker in text: text=text.split(marker,1)[0].strip()
    text=''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
    text=re.sub(r'(.)\1{4,}', r'\1\1', text)
    for n in range(1,min(16,len(text)//3)+1):
        u=text[-n:]
        if u and text.endswith(u*3): text=text[:-n*2].rstrip(); break
    return text
if __name__=="__main__": print(generate(input(">>> ")))
