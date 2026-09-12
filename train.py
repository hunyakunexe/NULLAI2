import json, random, re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SentencePieceTokenizer
from model import MiniLLM

MODEL_VERSION=8
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
CORPUS=ROOT/"data/pretrain.txt"; DATA=ROOT/"data/initial_ja.jsonl"; DISCORD=ROOT/"data/discord.jsonl"; SEED=ROOT/"data/chat_seed.jsonl"
CKPT=ROOT/"data/model.pt"; TOK=ROOT/"data/tokenizer.model"

def clean_text(s):
    s=str(s or "")
    s=re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s).replace("\ufffd", "")
    return s.strip()

def load_corpus():
    out=[]
    if CORPUS.exists():
        raw=clean_text(CORPUS.read_text(encoding="utf-8"))
        for d in re.split(r"<\|document\|>|<\|end_document\|>",raw):
            d=re.sub(r"^\s*-{4,}\s*$","",d,flags=re.M).strip()
            if len(d)>=80: out.append(d)
    return out

def load_chat_pairs():
    pairs=[]
    for p in (DATA,SEED,DISCORD):
        if not p.exists(): continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try: o=json.loads(line)
            except json.JSONDecodeError: continue
            inp=clean_text(o.get("input")); resp=clean_text(o.get("response"))
            if inp and resp: pairs.append((inp,resp))
    # Remove exact duplicates while keeping order.
    seen=set(); unique=[]
    for pair in pairs:
        if pair not in seen:
            seen.add(pair); unique.append(pair)
    return unique

def chat_prompt(inp):
    return f"Discord会話に自然に短く返信。\n会話:\n現在:{inp}\n経験:\n返信:"

class LMItemDataset(Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        x,y=self.items[i]
        return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)

def collate(batch):
    n=max(x.numel() for x,_ in batch); xs=[]; ys=[]
    for x,y in batch:
        xs.append(torch.cat((x,torch.zeros(n-x.numel(),dtype=torch.long))))
        ys.append(torch.cat((y,torch.full((n-y.numel(),),-100,dtype=torch.long))))
    return torch.stack(xs),torch.stack(ys)

def compatible(ck,tok,cfg):
    mc=ck.get("model_config",{})
    return ck.get("model_version")==MODEL_VERSION and ck.get("vocab_size")==tok.vocab_size and all(mc.get(k)==cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout"))

def run_epoch(model,loader,opt,device,epochs,label):
    model.train(); losses=[]
    accum=max(1,int(CFG.get("gradient_accumulation_steps",1)))
    amp=(device=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True); steps=0; epoch_losses=[]
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                _,loss,_=model(x,y)
            raw=float(loss.detach().cpu()); epoch_losses.append(raw); losses.append(raw)
            scaler.scale(loss/accum).backward(); steps+=1
            if steps%accum==0 or step==len(loader)-1:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        print(f"[train] {label} epoch={epoch+1}/{epochs} loss={sum(epoch_losses)/max(1,len(epoch_losses)):.4f}",flush=True)
    return losses

def train_once(incremental=False):
    seed=int(CFG.get("seed",42)); random.seed(seed); torch.manual_seed(seed)
    corpus=load_corpus(); pairs=load_chat_pairs()
    if not corpus and not pairs:
        print("[train] 学習データがありません。",flush=True); return False
    print(f"[train] corpus_docs={len(corpus)} chat_pairs={len(pairs)}",flush=True)
    # Keep the tokenizer stable during incremental learning. Rebuilding it would invalidate embeddings.
    cfg=dict(CFG); cfg.update({"vocab_size":int(CFG.get("vocab_size",8000)),"block_size":int(CFG.get("block_size",384))})
    old=None
    if CKPT.exists():
        try: old=torch.load(CKPT,map_location="cpu",weights_only=False)
        except Exception: old=None
    tokenizer_ready=TOK.exists() and old is not None and old.get("vocab_size") is not None
    if tokenizer_ready:
        tok=SentencePieceTokenizer(str(TOK))
        if int(old.get("vocab_size")) != tok.vocab_size: tokenizer_ready=False
    if not tokenizer_ready:
        tokenizer_texts=corpus+[chat_prompt(i)+r for i,r in pairs]
        tok=SentencePieceTokenizer.train(tokenizer_texts,TOK,int(CFG.get("vocab_size",8000)))
    tok.save_info(str(ROOT/"data/tokenizer.json"))
    cfg["vocab_size"]=tok.vocab_size
    model=MiniLLM(cfg)
    if old and compatible(old,tok,cfg): model.load_state_dict(old["model"])
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    do_pretrain=not (incremental and old and compatible(old,tok,cfg))

    # Stage 1: literary/general Japanese causal-LM pretraining.
    pre_items=[]
    for t in corpus:
        ids=tok.encode(t,add_bos=True,add_eos=True)
        for i in range(0,max(0,len(ids)-2),cfg["block_size"]):
            chunk=ids[i:i+cfg["block_size"]+1]
            if len(chunk)>=3: pre_items.append((chunk[:-1],chunk[1:]))
    if do_pretrain and pre_items:
        pre=DataLoader(LMItemDataset(pre_items),batch_size=int(CFG.get("batch_size",8)),shuffle=True,collate_fn=collate)
        opt=torch.optim.AdamW(model.parameters(),lr=float(CFG.get("learning_rate",3e-4)),weight_decay=float(CFG.get("weight_decay",.01)))
        run_epoch(model,pre,opt,device,int(CFG.get("pretrain_epochs",1)),"pretrain")

    # Stage 2: conversation fine-tuning. Loss is applied only to the assistant answer.
    chat_items=[]
    repeat=max(1,int(CFG.get("chat_sample_repeat",16)))
    for inp,resp in pairs:
        prompt=chat_prompt(inp)
        pids=tok.encode(prompt,add_bos=True,add_eos=False)
        rids=tok.sp.encode(resp,out_type=int)+[tok.eos_id]
        ids=pids+rids
        if len(ids)>cfg["block_size"]+1: continue
        labels=[-100]*len(pids)+rids
        for _ in range(repeat): chat_items.append((ids[:-1],labels[1:]))
    if chat_items:
        chat=DataLoader(LMItemDataset(chat_items),batch_size=max(1,min(int(CFG.get("batch_size",8)),len(chat_items))),shuffle=True,collate_fn=collate)
        opt=torch.optim.AdamW(model.parameters(),lr=float(CFG.get("chat_learning_rate",1e-4)),weight_decay=float(CFG.get("weight_decay",.01)))
        run_epoch(model,chat,opt,device,int(CFG.get("chat_epochs",4)),"chat")
    model.eval()
    CKPT.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":None,"model_version":MODEL_VERSION,"vocab_size":tok.vocab_size,
                "model_config":{k:cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout")},
                "tokens_seen":sum(len(tok.encode(t,add_bos=False,add_eos=False)) for t in corpus),"chat_pairs":len(pairs)},CKPT)
    print(f"[train] saved v{MODEL_VERSION}; corpus_docs={len(corpus)} chat_pairs={len(pairs)}",flush=True)
    return True

if __name__=="__main__": train_once(CKPT.exists())
