import asyncio,json,os,random,re,time
import torch
import discord
from experience_core import ExperienceCore
from semantic_engine import analyze
from tokenizer import SentencePieceTokenizer
from model import MiniLLM
try:
    from torch.ao.quantization import quantize_dynamic
except Exception:
    quantize_dynamic=None

class Bot(discord.Client):
    def __init__(self,cfg):
        super().__init__(intents=discord.Intents.all())
        self.cfg=cfg; self.core=ExperienceCore(decay_days=float(cfg.get('experience',{}).get('decay_days',60))); self.pending={}
        self.model=None; self.tok=None; self._train_lock=asyncio.Lock(); self._experience_count=0; self._maintenance_task=None; self._model_lock=asyncio.Lock()
        self._load_model()

    def _load_model(self):
        """Load the latest compatible checkpoint; safe on the very first boot."""
        try:
            tok=SentencePieceTokenizer('data/tokenizer.model')
            ck=torch.load('data/model.pt',map_location='cpu',weights_only=False)
            if ck.get('model_version') != 5:
                raise RuntimeError('checkpoint model_version is incompatible; startup maintenance will retrain it')
            mc=dict(self.cfg); mc['vocab_size']=tok.vocab_size
            saved=ck.get('model_config',{})
            for k in ('n_embd','n_head','n_layer','block_size','dropout'):
                if k in saved: mc[k]=saved[k]
            model=MiniLLM(mc)
            model.load_state_dict(ck['model'])
            model.eval()
            if self.cfg.get('inference',{}).get('quantized',True) and quantize_dynamic:
                model=quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8); model.eval()
            self.tok=tok; self.model=model
            print('[LLM] model loaded',flush=True)
            return True
        except Exception as e:
            self.tok=None; self.model=None
            print('[LLM] model unavailable yet:',e,flush=True)
            return False

    async def on_ready(self):
        print('READY',self.user,flush=True)
        if not getattr(self,'_maintenance_started',False):
            self._maintenance_started=True
            self._maintenance_task=asyncio.create_task(self._startup_maintenance())

    async def _startup_maintenance(self):
        print('[startup] maintenance task started',flush=True)
        async with self._train_lock:
            try:
                from startup_maintenance import run as run_startup_maintenance
                await asyncio.to_thread(run_startup_maintenance,self.cfg)
                self._load_model()  # first boot or post-training reload
                print('[startup] maintenance task finished',flush=True)
            except Exception:
                import traceback
                print('[startup] maintenance failed:',flush=True)
                traceback.print_exc()

    def allowed(self,m):
        p=self.cfg.get('privacy',{}); b=self.cfg.get('bot',{})
        if m.author.bot and not p.get('collect_bot_messages',False): return False
        if isinstance(m.channel,discord.DMChannel) and not p.get('collect_dms',False): return False
        if p.get('allowed_channel_ids') and str(m.channel.id) not in map(str,p['allowed_channel_ids']): return False
        if str(m.channel.id) in map(str,p.get('denied_channel_ids',[])): return False
        return True

    @staticmethod
    def _clean_reply(text):
        import unicodedata
        text=str(text or "").replace("\x00", "").strip()
        if not text: return ""
        # Cut generated prompt/control markers.
        for marker in ("\nユーザー:", "\nAI:", "\n現在:", "\n返信:", "\n回复:", "<|end_document|>", "<|document|>"):
            if marker in text:
                text=text.split(marker,1)[0].strip()
        # Remove Unicode control characters but keep normal Japanese punctuation/newlines.
        text=''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
        # Collapse whitespace without destroying Japanese text.
        text=re.sub(r'[ \t]{2,}', ' ', text)
        # Character-level runaway repetition: かなり/るるるるる etc.
        text=re.sub(r'(.)\1{4,}', r'\1\1', text)
        # Remove repeated suffixes of words/phrases.
        for unit_len in range(1, min(80, len(text)//3)+1):
            unit=text[-unit_len:]
            if unit and text.endswith(unit*3):
                text=text[:-unit_len*2].rstrip()
                break
        words=text.split()
        for n in range(1, min(16, len(words)//3)+1):
            if words[-n:]==words[-2*n:-n]==words[-3*n:-2*n]:
                text=' '.join(words[:-2*n]).strip(); break
        # Avoid output that is overwhelmingly punctuation/symbol noise.
        visible=[c for c in text if not c.isspace()]
        if visible:
            alnum=sum(c.isalnum() or ('\u3040'<=c<='\u30ff') or ('\u4e00'<=c<='\u9fff') for c in visible)
            if len(visible)>=12 and alnum/len(visible)<0.35:
                return ''
        # Keep the first few coherent sentences rather than a runaway paragraph.
        text=text.strip(' \t\n')
        return text

    def _generate_reply_sync(self,prompt):
        if not self.model or not self.tok: return ""
        ids=self.tok.encode(prompt,add_bos=True,add_eos=False)
        ids=ids[-self.model.block_size:]
        x=torch.tensor([ids],dtype=torch.long)
        inf=self.cfg.get('inference',{})
        with torch.no_grad():
            y=self.model.generate(x,max_new_tokens=int(inf.get('max_new_tokens',50)),temperature=float(inf.get('temperature',.55)),top_k=int(inf.get('top_k',24)),repetition_penalty=float(inf.get('repetition_penalty',1.28)),no_repeat_ngram_size=int(inf.get('no_repeat_ngram_size',3)),eos_token_id=self.tok.eos_id)
        return self._clean_reply(self.tok.decode(y[0].tolist()[len(ids):]).strip())

    def _fallback(self,a,ex):
        if ex and ex[0].get('response'):
            return self._clean_reply(ex[0]['response'])
        # Never use a content-like fixed phrase such as 「なるほど。」 as the normal fallback.
        return 'もう少し具体的に教えてください。' if a['semantic']=='question' else 'うまく返答を作れませんでした。'

    async def on_message(self,m):
        if not self.allowed(m) or m.author.bot: return
        parts=m.content.strip().split()
        if len(parts)>=2 and parts[0].lower()=='/ep' and parts[1].lower()=='channel':
            if not isinstance(m.author,discord.Member) or not m.author.guild_permissions.administrator:
                await m.channel.send('このコマンドは管理者専用です。'); return
            b=self.cfg.setdefault('bot',{}); ignored=b.setdefault('ignored_channel_ids',[])
            action=parts[2].lower() if len(parts)>=3 else 'list'
            if action=='list':
                if not ignored: await m.channel.send('返信しないチャンネルは設定されていません。'); return
                names=[]
                for cid in ignored:
                    ch=m.guild.get_channel(int(cid)) if m.guild else None
                    names.append(f'{ch.mention} (`{cid}`)' if ch else f'`{cid}`')
                await m.channel.send('返信しないチャンネル:\n'+'\n'.join(names)); return
            if action in ('deny','add','off'):
                cid=(parts[3] if len(parts)>=4 else str(m.channel.id)).strip('<>#&!')
                if not cid.isdigit(): await m.channel.send('チャンネルIDを指定してください。'); return
                if cid not in map(str,ignored): ignored.append(int(cid))
                with open('config.json','w',encoding='utf-8') as f: json.dump(self.cfg,f,ensure_ascii=False,indent=2)
                await m.channel.send(f'このチャンネル (`{cid}`) を返信対象外にしました。'); return
            if action in ('allow','remove','on'):
                cid=(parts[3] if len(parts)>=4 else str(m.channel.id)).strip('<>#&!')
                if not cid.isdigit(): await m.channel.send('チャンネルIDを指定してください。'); return
                b['ignored_channel_ids']=[x for x in ignored if str(x)!=cid]
                with open('config.json','w',encoding='utf-8') as f: json.dump(self.cfg,f,ensure_ascii=False,indent=2)
                await m.channel.send(f'このチャンネル (`{cid}`) を返信対象に戻しました。'); return
            await m.channel.send('使用法: `/ep channel deny [channel_id]` / `/ep channel allow [channel_id]` / `/ep channel list`'); return
        if str(m.channel.id) in map(str,self.cfg.get('bot',{}).get('ignored_channel_ids',[])): return
        if len(parts)>=2 and parts[0].lower()=='/ep' and parts[1].lower()=='probability':
            if not isinstance(m.author,discord.Member) or not m.author.guild_permissions.administrator:
                await m.channel.send('このコマンドは管理者専用です。'); return
            b=self.cfg.setdefault('bot',{})
            if len(parts)==2:
                await m.channel.send(f"通常メッセージへの割り込み返信確率: {float(b.get('occasional_reply_probability',0.08))*100:.1f}%"); return
            try: value=float(parts[2])
            except (ValueError,IndexError): await m.channel.send('使用法: /ep probability <0〜100>'); return
            if not 0<=value<=100: await m.channel.send('確率は0〜100%で指定してください。'); return
            b['occasional_reply_probability']=value/100.0
            with open('config.json','w',encoding='utf-8') as f: json.dump(self.cfg,f,ensure_ascii=False,indent=2)
            await m.channel.send(f'通常メッセージへの割り込み返信確率を {value:g}% に設定しました。'); return

        a=analyze(m.content,self.core.known())
        self.core.add_message(id=str(m.id),channel=str(m.channel.id),guild=str(m.guild.id) if m.guild else '',author=str(m.author.id),text=m.content,ts=m.created_at.timestamp())
        text=m.content.strip()
        q=re.match(r'^[「『]([^「」『』=]{1,80})[」』]\s*(?:=|は)\s*(.{1,300})$',text)
        if not q: q=re.match(r'^([^「」『』=\s]{1,80})=\s*(.{1,300})$',text)
        if q:
            self.core.add_term(q.group(1).strip(),q.group(2).strip())
            if self.user and self.user.mentioned_in(m): await m.channel.send('覚えました。')
            return
        b=self.cfg.get('bot',{}); mentioned=self.user and self.user.mentioned_in(m)
        if not mentioned:
            if b.get('reply_only_when_mentioned',False):
                if not b.get('occasional_reply',True) or random.random()>=float(b.get('occasional_reply_probability',0.08)): return
            elif not b.get('occasional_reply',True): return
        if not b.get('auto_reply',False): return
        ctx={'user':str(m.author.id),'channel':str(m.channel.id),'guild':str(m.guild.id) if m.guild else ''}
        ex=self.core.search(a['semantic'],ctx,int(b.get('candidate_limit',16)))
        if a['unknown']:
            reply=f'「{a["unknown"][0]}」ってどういう意味？'
        else:
            history=[x.content async for x in m.channel.history(limit=int(b.get('history_messages',12)),before=m)]
            prompt='Discord会話に自然に短く返信。\n会話:\n'+'\n'.join(reversed(history))+f'\n現在:{m.content}\n経験:\n'+''.join(f"{e['input']} -> {e['response']} [{e['result']}]\n" for e in ex[:8])+'返信:'
            reply=''
            if self.model and self.tok:
                try:
                    async with self._model_lock:
                        reply=await asyncio.to_thread(self._generate_reply_sync,prompt)
                except Exception as e:
                    print('[LLM] generation failed:',e,flush=True)
            if not reply: reply=self._fallback(a,ex)
        reply=self._clean_reply(reply) or self._fallback(a,ex)
        sent=await m.channel.send(reply[:int(b.get('max_reply_chars',1800))])
        self.pending[sent.id]=(m,a,reply,ctx)
        asyncio.create_task(self.observe(sent.id))

    async def _train_background(self):
        async with self._train_lock:
            try:
                from train import train_once
                changed=await asyncio.to_thread(train_once,True)
                if changed: self._load_model()
            except Exception as e: print('[train] background update failed:',e,flush=True)

    async def observe(self,sid):
        item=self.pending.pop(sid,None)
        if not item: return
        m,a,reply,ctx=item
        await asyncio.sleep(float(self.cfg.get('bot',{}).get('observation_delay_seconds',5)))
        msgs=[]
        async for x in m.channel.history(limit=int(self.cfg.get('bot',{}).get('observation_max_messages',8)),after=m):
            if x.id!=sid and not x.author.bot: msgs.append(x.content)
        pos=sum(any(z in t.lower() for z in ('いいね','それな','了解','わかる','うん','?','？','w','草')) for t in msgs)
        neg=sum(any(z in t for z in ('違う','いや','無理','ダメ')) for t in msgs)
        score=max(-1,min(1,(pos-neg)/max(1,len(msgs))))
        result='successful' if score>=.45 else 'failed' if score<=-.45 else 'neutral'
        self.core.add_experience(semantic=a['semantic'],input=m.content,context=ctx,response=reply,reactions=msgs,next_messages=msgs,result=result,score=score,confidence=.7 if msgs else .45,fingerprint=a['fingerprint'])
        if result=='successful' or score>=float(self.cfg.get('experience',{}).get('export_score',.45)):
            self.core.export(min_score=float(self.cfg.get('experience',{}).get('export_score',.45)),min_conf=float(self.cfg.get('experience',{}).get('export_confidence',.65)))
            self._experience_count+=1
            every=int(self.cfg.get('training',{}).get('train_every_experiences',20))
            if every>0 and self._experience_count%every==0:
                asyncio.create_task(self._train_background())
