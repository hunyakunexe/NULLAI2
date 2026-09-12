import asyncio,json,os,random,re,time
import torch
import discord
from experience_core import ExperienceCore
from social_memory import SocialMemory
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
        self.cfg=cfg; self.core=ExperienceCore(decay_days=float(cfg.get('experience',{}).get('decay_days',60))); self.social=SocialMemory(decay_days=float(cfg.get('social_memory',{}).get('decay_days',45)),retention_days=float(cfg.get('social_memory',{}).get('retention_days',180))) if cfg.get('social_memory',{}).get('enabled',True) else None; self.pending={}
        self.model=None; self.tok=None; self._train_lock=asyncio.Lock(); self._experience_count=0; self._maintenance_task=None; self._model_lock=asyncio.Lock()
        self._load_model()

    def _load_model(self):
        """Load the latest compatible checkpoint; safe on the very first boot."""
        try:
            tok=SentencePieceTokenizer('data/tokenizer.model')
            ck=torch.load('data/model.pt',map_location='cpu',weights_only=False)
            if ck.get('model_version') != 8:
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
        if not text:return ""
        # Never allow the model to echo its own prompt/control fields.
        text=re.sub(r"^(?:ユーザー|利用者|AI|アシスタント|現在|返信|回答|assistant|user)\s*[:：]\s*", "", text, flags=re.I).strip()
        text=re.split(r"\n(?:ユーザー|利用者|AI|アシスタント|現在|返信|回答|assistant|user)\s*[:：]",text,maxsplit=1,flags=re.I)[0].strip()
        for marker in ("<|end_document|>","<|document|>","<|eos|>"):
            if marker in text:text=text.split(marker,1)[0].strip()
        text=''.join(ch for ch in text if unicodedata.category(ch)[0]!='C' or ch in '\n\t')
        text=re.sub(r'[ \t]{2,}',' ',text).strip()
        # Collapse character and phrase loops, including loops in the middle.
        text=re.sub(r'(.)\1{4,}',r'\1\1',text)
        for n in range(1,min(30,len(text)//3)+1):
            pat=re.escape(text[:n])
            if re.fullmatch(r'(?:'+pat+r'){3,}',text):return text[:n].strip()
        # Remove a repeated n-gram when it occupies a large part of the answer.
        for n in range(2,min(24,len(text)//3)+1):
            found=False
            for i in range(0,len(text)-3*n+1):
                unit=text[i:i+n]
                if unit and text[i+2*n:i+3*n]==unit and text[i+n:i+2*n]==unit:
                    text=text[:i+n]+text[i+3*n:]
                    found=True;break
            if found:break
        text=text.strip(' \t\n')
        visible=[c for c in text if not c.isspace()]
        if not visible:return ''
        jp=sum(c.isalnum() or ('\u3040'<=c<='\u30ff') or ('\u4e00'<=c<='\u9fff') for c in visible)
        if len(visible)>=10 and jp/len(visible)<0.45:return ''
        if len(set(visible))<=2 and len(visible)>=5:return ''
        return text

    def _generate_reply_sync(self,prompt):
        if not self.model or not self.tok: return ""
        ids=self.tok.encode(prompt,add_bos=True,add_eos=False)
        ids=ids[-self.model.block_size:]
        x=torch.tensor([ids],dtype=torch.long)
        inf=self.cfg.get('inference',{})
        with torch.no_grad():
            y=self.model.generate(x,max_new_tokens=int(inf.get('max_new_tokens',60)),temperature=float(inf.get('temperature',.42)),top_k=int(inf.get('top_k',16)),repetition_penalty=float(inf.get('repetition_penalty',1.18)),no_repeat_ngram_size=int(inf.get('no_repeat_ngram_size',3)),eos_token_id=self.tok.eos_id)
        text=self._clean_reply(self.tok.decode(y[0].tolist()[len(ids):]).strip())
        if self._looks_bad(text):
            with torch.no_grad():
                y=self.model.generate(x,max_new_tokens=min(40,int(inf.get('max_new_tokens',60))),temperature=0.18,top_k=1,repetition_penalty=1.25,no_repeat_ngram_size=3,eos_token_id=self.tok.eos_id)
            text=self._clean_reply(self.tok.decode(y[0].tolist()[len(ids):]).strip())
        return text

    @staticmethod
    def _looks_bad(text):
        if not text or len(text)<2: return True
        visible=[c for c in text if not c.isspace()]
        if len(visible)<2: return True
        # Reject obvious token/gibberish runs before falling back to a natural response.
        if re.search(r'(.)\1{3,}',text): return True
        jp=sum(('\u3040'<=c<='\u30ff') or ('\u4e00'<=c<='\u9fff') or c.isalnum() for c in visible)
        if len(visible)>=8 and jp/len(visible)<0.45: return True
        if len(set(visible))<=2 and len(visible)>=5: return True
        return False

    @staticmethod
    def _best_experience(text,ex):
        import difflib
        norm=lambda x:re.sub(r"[\s　]+","",str(x or "")).lower()
        target=norm(text)
        if not target:return ''
        best=None;best_score=0.0
        for e in ex or []:
            resp=str(e.get('response') or '').strip()
            inp=norm(e.get('input',''))
            if not resp or not inp:continue
            sim=difflib.SequenceMatcher(None,target,inp).ratio()
            if inp==target:sim=1.0
            score=sim*(0.75+0.25*min(1,float(e.get('confidence',0.0))))
            if e.get('result')=='failed':score*=0.35
            if score>best_score:best_score=score;best=resp
        return best if best_score>=0.78 else ''

    def _fallback(self,a,ex,current_text=''):
        best=self._best_experience(current_text,ex)
        if best:return self._clean_reply(best)
        # Use a small natural seed set instead of exposing an internal failure message.
        seeds={
            'question':['その点は気になるね。もう少し具体的に教えてくれたら、一緒に考えられるよ。','なるほど。そこはもう少し詳しく聞いてみたい。'],
            'statement':['なるほど、それは面白そうだね。もう少し詳しく聞かせて。','それは気になる。続きがあれば聞きたいな。'],
            '挨拶':['こんにちは。今日はどうしたの？','お疲れさま。今日はどんな話をしようか？'],
            '感謝':['どういたしまして。役に立てたならよかった。','いえいえ、こちらこそ。'],
        }
        return random.choice(seeds.get(a.get('semantic'),'statement'))

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
        guild_id=str(m.guild.id) if m.guild else ''
        mentioned_ids=[str(u.id) for u in getattr(m,'mentions',[]) if getattr(u,'id',None) is not None]
        replied_user_id=None
        try:
            ref=getattr(m,'reference',None)
            resolved=getattr(ref,'resolved',None) if ref else None
            if resolved is not None and getattr(resolved,'author',None) is not None:
                replied_user_id=str(resolved.author.id)
        except Exception:
            replied_user_id=None
        self.core.add_message(id=str(m.id),channel=str(m.channel.id),guild=guild_id,author=str(m.author.id),text=m.content,ts=m.created_at.timestamp())
        if self.social:
            self.social.record_message(guild_id=guild_id,channel_id=str(m.channel.id),user_id=str(m.author.id),display_name=getattr(m.author,'display_name',getattr(m.author,'name','')) if self.cfg.get('social_memory',{}).get('store_display_names',True) else '',
                                   text=m.content,concept=a.get('semantic','その他'),mentioned_user_ids=mentioned_ids,replied_user_id=replied_user_id,
                                   ts=m.created_at.timestamp(),bot_user_id=str(self.user.id) if self.user else None)
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
        recent_users=[]
        try:
            hist_objs=[x async for x in m.channel.history(limit=int(b.get('history_messages',12)),before=m)]
            recent_users=[str(x.author.id) for x in hist_objs]
            history=[x.content for x in reversed(hist_objs)]
        except Exception:
            hist_objs=[]; history=[]
        ctx={'user':str(m.author.id),'channel':str(m.channel.id),'guild':guild_id}
        social_ctx=self.social.context_for(guild_id,str(m.author.id),str(self.user.id) if self.user else None,recent_users,str(m.channel.id)) if self.social and self.cfg.get('social_memory',{}).get('include_in_prompt',True) else {}
        ctx['social']=social_ctx
        ex=self.core.search(a['semantic'],ctx,int(b.get('candidate_limit',16)))
        if a['unknown']:
            reply=f'「{a["unknown"][0]}」ってどういう意味？'
        else:
            exact_reply=self._best_experience(m.content,ex)
            if exact_reply:
                reply=exact_reply
            else:
                history_lines=[]
                for obj in reversed(hist_objs):
                    name=getattr(obj.author,'display_name',getattr(obj.author,'name','ユーザー'))
                    history_lines.append(f'{name}: {obj.content}')
                prompt=('Discord会話に自然な日本語で短く返信する。\n'
                         '相手の発言に直接反応し、無関係な説明を足さない。\n'
                         '意味のない繰り返し、同じ語の連続、学習データの引用、統計の説明はしない。\n'
                         '会話:\n'+'\n'.join(history_lines)+f'\n現在: {m.content}\n'
                         +self.social.prompt_text(social_ctx)+'\n経験（参考。内容をそのまま引用しない）:\n'
                         +''.join(f"入力: {e['input']}\n応答例: {e['response']}\n" for e in ex[:5])+'返信:')
                reply=''
                if self.model and self.tok:
                    try:
                        async with self._model_lock:
                            reply=await asyncio.to_thread(self._generate_reply_sync,prompt)
                    except Exception as e:
                        print('[LLM] generation failed:',e,flush=True)
                if not reply: reply=self._fallback(a,ex,m.content)
        reply=self._clean_reply(reply) or self._fallback(a,ex,m.content)
        sent=await m.channel.send(reply[:int(b.get('max_reply_chars',1800))])
        self.pending[sent.id]=(m,a,reply,ctx)
        if self.social and guild_id and self.user:
            self.social.observe_bot_result(guild_id,str(m.author.id),str(self.user.id),ts=time.time())
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
        msgs=[]; direct=[]
        async for x in m.channel.history(limit=int(self.cfg.get('bot',{}).get('observation_max_messages',8)),after=m):
            if x.id==sid or x.author.bot: continue
            msgs.append(x.content)
            ref=getattr(x,'reference',None)
            resolved=getattr(ref,'resolved',None) if ref else None
            if resolved is not None and getattr(resolved,'id',None)==sid: direct.append(x.content)
            elif self.user and any(getattr(u,'id',None)==self.user.id for u in getattr(x,'mentions',[])): direct.append(x.content)
        def sentiment(t):
            low=t.lower(); pos=any(z in low for z in ('いいね','それな','了解','わかる','助か','ありがとう','うん','w','草')); neg=any(z in t for z in ('違う','いや','無理','ダメ','わからない'))
            return (1 if pos else 0)-(1 if neg else 0)
        # Direct replies/mentions are much stronger evidence than unrelated channel traffic.
        dscore=sum(sentiment(t) for t in direct)/max(1,len(direct)) if direct else 0.0
        nscore=sum(sentiment(t) for t in msgs)/max(1,len(msgs)) if msgs else 0.0
        score=max(-1,min(1,dscore*.75+nscore*.25))
        result='successful' if score>=.45 else 'failed' if score<=-.45 else 'neutral'
        if self.social and ctx.get('guild') and self.user:
            self.social.observe_bot_result(ctx['guild'],str(m.author.id),str(self.user.id),positive=result=='successful',negative=result=='failed',ts=time.time())
        self.core.add_experience(semantic=a['semantic'],input=m.content,context=ctx,response=reply,reactions=msgs,next_messages=msgs,result=result,score=score,confidence=.7 if msgs else .45,fingerprint=a['fingerprint'])
        if result=='successful' or score>=float(self.cfg.get('experience',{}).get('export_score',.45)):
            self.core.export(min_score=float(self.cfg.get('experience',{}).get('export_score',.45)),min_conf=float(self.cfg.get('experience',{}).get('export_confidence',.65)))
            self._experience_count+=1
            every=int(self.cfg.get('training',{}).get('train_every_experiences',20))
            if every>0 and self._experience_count%every==0:
                asyncio.create_task(self._train_background())
