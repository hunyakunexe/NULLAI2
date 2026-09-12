import json, random, re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SentencePieceTokenizer
from model import MiniLLM

MODEL_VERSION=5
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
CORPUS=ROOT/"data/pretrain.txt"; DATA=ROOT/"data/initial_ja.jsonl"; DISCORD=ROOT/"data/discord.jsonl"
CKPT=ROOT/"data/model.pt"; TOK=ROOT/"data/tokenizer.model"

def clean_text(s):
    s=str(s or "")
    s=re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s).replace("\ufffd","")
    return s.strip()

def load_texts():
    out=[]
    if CORPUS.exists():
        raw=clean_text(CORPUS.read_text(encoding="utf-8"))
        docs=re.split(r"<\|document\|>|<\|end_document\|>",raw)
        for d in docs:
            d=re.sub(r"^\s*-{4,}\s*$","",d,flags=re.M).strip()
            if len(d)>=80: out.append(d)
    for p in (DATA,DISCORD):
        if not p.exists(): continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                o=json.loads(line)
            except json.JSONDecodeError:
                continue
            inp=clean_text(o.get("input"))
            resp=clean_text(o.get("response"))
            text=clean_text(o.get("text"))
            if inp and resp:
                sample=f"ユーザー: {inp}\nアシスタント: {resp}"
                out.extend([sample]*max(1,int(CFG.get("chat_sample_repeat",8))))
            elif text:
                out.append(text)
    return out

class ChunkDataset(Dataset):
    def __init__(self, sequences, block_size):
        self.items=[]
        for ids in sequences:
            if len(ids)<3: continue
            for i in range(0,len(ids)-1,block_size):
                chunk=ids[i:i+block_size+1]
                if len(chunk)>=3: self.items.append(chunk)
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        a=self.items[i]
        return torch.tensor(a[:-1],dtype=torch.long),torch.tensor(a[1:],dtype=torch.long)

def collate(batch):
    n=max(x.numel() for x,_ in batch)
    xs=[]; ys=[]
    for x,y in batch:
        xs.append(torch.cat((x,torch.zeros(n-x.numel(),dtype=torch.long))))
        ys.append(torch.cat((y,torch.full((n-y.numel(),),-100,dtype=torch.long))))
    return torch.stack(xs),torch.stack(ys)

def compatible(ck,tok,cfg):
    mc=ck.get("model_config",{})
    return ck.get("model_version")==MODEL_VERSION and ck.get("vocab_size")==tok.vocab_size and all(mc.get(k)==cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout"))

def train_once(incremental=False):
    random.seed(int(CFG.get("seed",42))); torch.manual_seed(int(CFG.get("seed",42)))
    texts=load_texts()
    if not texts:
        print("[train] 学習データがありません。",flush=True); return False
    print(f"[train] loaded {len(texts)} records",flush=True)

    # Rebuild tokenizer for v5 so the Aozora corpus and chat data share one clean vocabulary.
    tok=SentencePieceTokenizer.train(texts,TOK,int(CFG.get("vocab_size",8000)))
    tok.save_info(str(ROOT/"data/tokenizer.json"))
    cfg=dict(CFG); cfg.update({"vocab_size":tok.vocab_size,"block_size":int(CFG.get("block_size",384))})
    model=MiniLLM(cfg)

    old=None
    if CKPT.exists():
        try: old=torch.load(CKPT,map_location="cpu",weights_only=False)
        except Exception: old=None
    if old and compatible(old,tok,cfg):
        model.load_state_dict(old["model"])

    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device).train()
    seqs=[tok.encode(t) for t in texts]
    ds=ChunkDataset(seqs,cfg["block_size"])
    if not ds: return False
    loader=DataLoader(ds,batch_size=int(CFG.get("batch_size",8)),shuffle=True,collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(),lr=float(CFG.get("learning_rate",3e-4)),weight_decay=float(CFG.get("weight_decay",.01)))
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    grad_acc=max(1,int(CFG.get("gradient_accumulation_steps",1))); losses=[]
    for epoch in range(int(CFG.get("epochs_per_update",1))):
        opt.zero_grad(set_to_none=True)
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device=="cuda"):
                _,loss,_=model(x,y)
            scaler.scale(loss/grad_acc).backward()
            if (step+1)%grad_acc==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        # Flush the final partial accumulation.
        if len(loader)%grad_acc:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    CKPT.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"model_version":MODEL_VERSION,
                "vocab_size":tok.vocab_size,"model_config":{k:cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout")},
                "tokens_seen":sum(map(len,seqs))},CKPT)
    print(f"[train] saved v{MODEL_VERSION}; records={len(texts)} chunks={len(ds)} avg_loss={sum(losses)/max(1,len(losses)):.4f}",flush=True)
    return True

if __name__=="__main__":
    train_once(CKPT.exists())
