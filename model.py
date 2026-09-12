import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        if n_embd % n_head:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1,1,block_size,block_size), persistent=False)

    def forward(self, x, past_kv=None, use_cache=False):
        B,T,C=x.shape
        q,k,v=self.qkv(x).split(C,dim=2)
        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        past_len=0
        if past_kv is not None:
            pk,pv=past_kv
            past_len=pk.size(-2)
            k=torch.cat((pk,k),dim=-2)
            v=torch.cat((pv,v),dim=-2)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if past_kv is None:
            att=att.masked_fill(~self.mask[:,:,:T,:T],float("-inf"))
        elif T > 1:
            key_len=k.size(-2)
            causal=torch.tril(torch.ones(T,key_len,device=x.device,dtype=torch.bool), diagonal=past_len)
            att=att.masked_fill(~causal[None,None],float("-inf"))
        att=F.softmax(att,dim=-1)
        y=att@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        cache=(k,v) if use_cache else None
        return self.drop(self.proj(y)),cache

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg["n_embd"])
        self.attn=CausalSelfAttention(cfg["n_embd"],cfg["n_head"],cfg["block_size"],cfg["dropout"])
        self.ln2=nn.LayerNorm(cfg["n_embd"])
        self.mlp=nn.Sequential(nn.Linear(cfg["n_embd"],4*cfg["n_embd"]),nn.GELU(),nn.Linear(4*cfg["n_embd"],cfg["n_embd"]),nn.Dropout(cfg["dropout"]))

    def forward(self,x,past_kv=None,use_cache=False):
        a,c=self.attn(self.ln1(x),past_kv,use_cache)
        x=x+a
        x=x+self.mlp(self.ln2(x))
        return x,c

class MiniLLM(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.block_size=int(cfg["block_size"])
        self.tok=nn.Embedding(cfg["vocab_size"],cfg["n_embd"])
        self.pos=nn.Embedding(self.block_size,cfg["n_embd"])
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln=nn.LayerNorm(cfg["n_embd"])
        self.head=nn.Linear(cfg["n_embd"],cfg["vocab_size"],bias=False)
        self.head.weight=self.tok.weight

    def forward(self,idx,targets=None,past_kv=None,use_cache=False):
        B,T=idx.shape
        past_len=0 if past_kv is None else past_kv[0][0].size(-2)
        available=self.block_size-past_len
        if available <= 0:
            raise ValueError("KV cache is already at block_size; rebuild the cache before calling forward")
        if T>available:
            idx=idx[:,-available:]
            T=idx.size(1)
        pos=torch.arange(past_len,past_len+T,device=idx.device)
        x=self.tok(idx)+self.pos(pos)[None,:,:]
        new_cache=[]
        for block,past in zip(self.blocks,past_kv or [None]*len(self.blocks)):
            x,c=block(x,past,use_cache)
            if use_cache:
                new_cache.append(c)
        logits=self.head(self.ln(x))
        loss=None
        if targets is not None:
            if targets.shape[-1] != T:
                targets=targets[...,-T:]
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100)
        return logits,loss,(new_cache if use_cache else None)

    @torch.no_grad()
    def generate(self,idx,max_new_tokens=100,temperature=.75,top_k=30,repetition_penalty=1.15,no_repeat_ngram_size=3,eos_token_id=None,repeat_stop=3):
        """Generate safely; idx must be a prompt without a trailing EOS token."""
        if idx.ndim != 2 or idx.size(0) != 1:
            raise ValueError("generate currently expects a single prompt with shape [1, T]")
        max_new_tokens=max(0,int(max_new_tokens))
        if max_new_tokens == 0:
            return idx
        cache=None
        cur=idx[:,-self.block_size:]
        logits,_,cache=self(cur,use_cache=True)
        generated=[]
        for _ in range(max_new_tokens):
            logits=logits[:,-1,:]/max(float(temperature),1e-5)
            if repetition_penalty and float(repetition_penalty)>1.0 and generated:
                # Penalize only tokens generated for this answer, not the entire prompt.
                for token_id in set(generated):
                    score=logits[0,token_id]
                    penalty=float(repetition_penalty)
                    logits[0,token_id]=score/penalty if score>0 else score*penalty
            n=int(no_repeat_ngram_size)
            if n>=2 and len(generated)>=n-1:
                # Block an n-gram that has already appeared in the generated answer.
                prefix=tuple(generated[-(n-1):])
                seen=set()
                for i in range(len(generated)-n+1):
                    if tuple(generated[i:i+n-1])==prefix:
                        seen.add(generated[i+n-1])
                for token_id in seen:
                    logits[0,token_id]=-float("inf")
            if top_k and int(top_k)>0:
                k=min(int(top_k),logits.size(-1))
                v,_=torch.topk(logits,k)
                logits=logits.masked_fill(logits<v[:,-1,None],-float("inf"))
            probs=F.softmax(logits,dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum())<=0:
                probs=torch.zeros_like(logits).softmax(dim=-1)
            nxt=torch.multinomial(probs,1)
            token=int(nxt.item())
            cur=torch.cat((cur,nxt),1)
            generated.append(token)
            if eos_token_id is not None and token==int(eos_token_id):
                break
            # Hard-stop exact token cycles even when the model refuses to emit EOS.
            r=max(2,int(repeat_stop))
            stopped=False
            for width in range(1,min(12,len(generated)//r)+1):
                tail=generated[-width*r:]
                if len(tail)==width*r and all(tail[i*width:(i+1)*width]==tail[:width] for i in range(1,r)):
                    stopped=True
                    break
            if stopped:
                break
            if cache and cache[0][0].size(-2)>=self.block_size:
                cur=cur[:,-self.block_size:]
                logits,_,cache=self(cur,use_cache=True)
            else:
                logits,_,cache=self(nxt,use_cache=True,past_kv=cache)
        return cur
