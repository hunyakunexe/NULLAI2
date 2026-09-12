import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SentencePieceTokenizer
from model import MiniLLM

CFG=json.load(open("config.json",encoding="utf-8"))
CORPUS=Path("data/pretrain.txt")
DATA=Path("data/initial_ja.jsonl")
DISCORD=Path("data/discord.jsonl")
CKPT=Path("data/model.pt")
TOK=Path("data/tokenizer.model")
MODEL_VERSION=3

def load_sources():
    tokenizer_texts=[]
    training_texts=[]
    if CORPUS.exists():
        t=CORPUS.read_text(encoding="utf-8").strip()
        if t:
            tokenizer_texts.append(t)
            training_texts.append(t)
    if DATA.exists():
        for line in DATA.read_text(encoding="utf-8").splitlines():
            try:
                o=json.loads(line)
            except json.JSONDecodeError:
                continue
            text=str(o.get("text") or "").strip()
            if text:
                tokenizer_texts.append(text)
                # Teach the model the exact response position used at inference.
                sample="Discord会話に自然に短く返信。\n返信:"+text
                training_texts.extend([sample] * max(1, int(CFG.get("chat_sample_repeat", 4))))
    if DISCORD.exists():
        for line in DISCORD.read_text(encoding="utf-8").splitlines():
            try:
                o=json.loads(line)
            except json.JSONDecodeError:
                continue
            inp=str(o.get("input") or "").strip()
            resp=str(o.get("response") or "").strip()
            if inp and resp:
                sample="Discord会話に自然に短く返信。\n現在:"+inp+"\n返信:"+resp
                tokenizer_texts.extend([inp,resp,sample])
                training_texts.extend([sample] * max(1, int(CFG.get("chat_sample_repeat", 4))))
            else:
                text=str(o.get("text") or "").strip()
                if text:
                    tokenizer_texts.append(text); training_texts.append(text)
    return tokenizer_texts,training_texts

def load_texts():
    return load_sources()[1]

class ChunkDataset(Dataset):
    def __init__(self,sequences,block_size):
        self.items=[]
        for ids in sequences:
            if len(ids)<3: continue
            for i in range(0,len(ids)-1,block_size):
                chunk=ids[i:i+block_size+1]
                if len(chunk)>=3:
                    self.items.append(chunk)
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        a=self.items[i]
        return torch.tensor(a[:-1],dtype=torch.long),torch.tensor(a[1:],dtype=torch.long)

def Fpad(x,n,p):
    if x.numel()>=n:return x
    return torch.cat((x,torch.full((n-x.numel(),),p,dtype=x.dtype)))

def collate(batch):
    n=max(x[0].numel() for x in batch)
    xs=[Fpad(x,n,0) for x,_ in batch]
    ys=[Fpad(y,n,-100) for _,y in batch]
    return torch.stack(xs),torch.stack(ys)

def _checkpoint_matches(old,tok,cfg):
    mc=old.get("model_config",{})
    return (old.get("model_version")==MODEL_VERSION and old.get("vocab_size")==tok.vocab_size and
            all(mc.get(k)==cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout")))

def train_once(incremental=False):
    torch.manual_seed(int(CFG.get("seed",42)))
    print(f"[train] starting incremental={incremental}",flush=True)
    tokenizer_texts,texts=load_sources()
    if not texts:
        print("[train] 学習データがありません。",flush=True); return False
    if not TOK.exists():
        tok=SentencePieceTokenizer.train(tokenizer_texts,TOK,int(CFG.get("vocab_size",8000)))
        tok.save_info("data/tokenizer.json")
    else:
        tok=SentencePieceTokenizer(TOK)
    print(f"[train] tokenizer ready vocab_size={tok.vocab_size}",flush=True)
    cfg=dict(CFG); cfg.update({"vocab_size":tok.vocab_size,"block_size":int(CFG.get("block_size",256))})
    model=MiniLLM(cfg)
    old=None
    if CKPT.exists():
        try: old=torch.load(CKPT,map_location="cpu",weights_only=False)
        except Exception as e: print(f"[train] checkpoint load skipped: {e}",flush=True)
    compatible=old is not None and _checkpoint_matches(old,tok,cfg)
    if compatible:
        model.load_state_dict(old["model"])
    else:
        if old is not None: print("[train] old/incompatible checkpoint ignored; starting a clean model",flush=True)
        incremental=False
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.train()
    seqs=[tok.encode(t) for t in texts]
    ds=ChunkDataset(seqs,cfg["block_size"])
    if not ds:
        print("[train] 学習可能なチャンクがありません。",flush=True); return False
    bs=max(1,int(CFG.get("batch_size",8)))
    loader=DataLoader(ds,batch_size=bs,shuffle=True,collate_fn=collate)
    print(f"[train] texts={len(texts)} chunks={len(ds)} device={device}",flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=float(CFG.get("learning_rate",3e-4)),weight_decay=float(CFG.get("weight_decay",.01)))
    if compatible and old.get("optimizer"):
        try: opt.load_state_dict(old["optimizer"])
        except Exception: print("[train] optimizer state incompatible; using a fresh optimizer",flush=True)
    scaler=torch.amp.GradScaler("cuda",enabled=(device=="cuda"))
    losses=[]; grad_acc=max(1,int(CFG.get("gradient_accumulation_steps",1))); epochs=max(1,int(CFG.get("epochs_per_update",1)))
    log_every=max(1,int(CFG.get("log_every_steps",200)))
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        for step,(x,y) in enumerate(loader,1):
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device=="cuda")):
                _,loss,_=model(x,y)
            scaler.scale(loss/grad_acc).backward()
            if step%grad_acc==0 or step==len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if step%log_every==0 or step==len(loader):
                print(f"[train] epoch={epoch+1}/{epochs} step={step}/{len(loader)} loss={losses[-1]:.4f}",flush=True)
    CKPT.parent.mkdir(exist_ok=True,parents=True)
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"model_version":MODEL_VERSION,
                "vocab_size":tok.vocab_size,"model_config":{k:cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout")},
                "tokens_seen":sum(len(s) for s in seqs)},CKPT)
    print(f"[train] {'incremental' if incremental else 'initial'} avg_loss={sum(losses)/max(1,len(losses)):.4f} device={device}",flush=True)
    return True

def main(): train_once(CKPT.exists())
if __name__=="__main__": main()
