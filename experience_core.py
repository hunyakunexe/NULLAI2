import sqlite3,time,json,math,re,difflib
from pathlib import Path

class ExperienceCore:
    def __init__(self,path='data/experience.db',decay_days=60):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.decay_days=max(1.0,float(decay_days))
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.init()

    def init(self):
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS experiences(
          id INTEGER PRIMARY KEY,semantic TEXT,input TEXT,context TEXT,response TEXT,
          reactions TEXT,next_messages TEXT,result TEXT,score REAL,frequency INTEGER,
          confidence REAL,ts REAL,fingerprint TEXT);
        CREATE INDEX IF NOT EXISTS ex_sem ON experiences(semantic);
        CREATE INDEX IF NOT EXISTS ex_fp ON experiences(fingerprint);
        CREATE TABLE IF NOT EXISTS terms(term TEXT PRIMARY KEY,meaning TEXT,semantic TEXT,confidence REAL,source TEXT,ts REAL);
        CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,channel TEXT,guild TEXT,author TEXT,text TEXT,ts REAL);
        CREATE INDEX IF NOT EXISTS msg_channel ON messages(channel);
        CREATE TABLE IF NOT EXISTS profiles(kind TEXT,key TEXT PRIMARY KEY,stats TEXT,ts REAL);
        ''')
        self.db.commit()

    def add_term(self,t,m,s='learned',c=.9,source='human'):
        t=str(t).strip(); m=str(m).strip()
        if not t or not m:return
        self.db.execute('INSERT OR REPLACE INTO terms VALUES(?,?,?,?,?,?)',(t,m,s,float(c),source,time.time()));self.db.commit()
    def known(self): return {r['term'] for r in self.db.execute('SELECT term FROM terms')}
    def add_message(self,**x):
        self.db.execute('INSERT OR REPLACE INTO messages VALUES(?,?,?,?,?,?)',(str(x['id']),x.get('channel'),x.get('guild'),x.get('author'),x.get('text'),x.get('ts',time.time())));self.db.commit()

    def add_experience(self,**x):
        context=json.dumps(x.get('context',{}),ensure_ascii=False,sort_keys=True)
        fp=x.get('fingerprint','')
        row=self.db.execute('SELECT id,frequency FROM experiences WHERE fingerprint=? AND input=? AND response=? AND context=? ORDER BY id DESC LIMIT 1',(fp,x['input'],x['response'],context)).fetchone()
        vals=(x['semantic'],json.dumps(x.get('reactions',[]),ensure_ascii=False),json.dumps(x.get('next_messages',[]),ensure_ascii=False),x.get('result','neutral'),float(x.get('score',0)),float(x.get('confidence',0)),time.time())
        if row:
            self.db.execute('UPDATE experiences SET semantic=?,reactions=?,next_messages=?,result=?,score=?,frequency=?,confidence=?,ts=? WHERE id=?',(*vals,int(row['frequency'])+1,row['id']))
        else:
            self.db.execute('INSERT INTO experiences(semantic,input,context,response,reactions,next_messages,result,score,frequency,confidence,ts,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(x['semantic'],x['input'],context,x['response'],vals[1],vals[2],vals[3],vals[4],1,vals[5],vals[6],fp))
        self.db.commit()

    @staticmethod
    def _norm(s):
        return re.sub(r'[\s　]+','',str(s or '').lower())
    @classmethod
    def _similarity(cls,a,b):
        a,b=cls._norm(a),cls._norm(b)
        if not a or not b:return 0.0
        return difflib.SequenceMatcher(None,a,b).ratio()

    def search(self,semantic,ctx,limit=16):
        rows=self.db.execute('SELECT * FROM experiences WHERE semantic=? ORDER BY ts DESC LIMIT ?', (semantic,max(1,int(limit)*12))).fetchall()
        now=time.time(); out=[]
        current_input=ctx.get('input','') if isinstance(ctx,dict) else ''
        for r in rows:
            try:c=json.loads(r['context'])
            except Exception:c={}
            age=max(0,(now-r['ts'])/86400.0); decay=math.exp(-age/self.decay_days)
            exact=float(self._norm(r['input'])==self._norm(current_input)) if current_input else 0.0
            sim=self._similarity(r['input'],current_input) if current_input else 0.0
            match=(.18*(c.get('channel')==ctx.get('channel')) + .12*(c.get('guild')==ctx.get('guild')) + .12*(c.get('user')==ctx.get('user')))
            quality=max(-1,min(1,float(r['score'])))
            score=(quality*.48+float(r['confidence'])*.18+min(1,int(r['frequency'])/12)*.10+sim*.18+exact*.22+match)*decay
            out.append((score,dict(r)))
        return [x[1] for x in sorted(out,key=lambda x:x[0],reverse=True)[:max(1,int(limit))]]

    def export(self,path='data/curated_experiences.jsonl',min_score=.45,min_conf=.65,discord_path='data/discord.jsonl'):
        rows=self.db.execute('SELECT * FROM experiences WHERE score>=? AND confidence>=? ORDER BY ts ASC',(float(min_score),float(min_conf))).fetchall()
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(discord_path).parent.mkdir(parents=True,exist_ok=True)
        with open(path,'w',encoding='utf8') as f, open(discord_path,'w',encoding='utf8') as d:
            for r in rows:
                try:ctx=json.loads(r['context'])
                except Exception:ctx={}
                f.write(json.dumps({'instruction':'Discord会話に自然に返信する','input':r['input'],'context':ctx,'output':r['response'],'result':r['result'],'score':r['score']},ensure_ascii=False)+'\n')
                d.write(json.dumps({'input':r['input'],'response':r['response'],'result':r['result'],'score':r['score'],'confidence':r['confidence'],'timestamp':r['ts']},ensure_ascii=False)+'\n')
        return len(rows)

    def forget_channel(self,ch):
        ch=str(ch); self.db.execute('DELETE FROM messages WHERE channel=?',(ch,))
        ids=[]
        for r in self.db.execute('SELECT id,context FROM experiences').fetchall():
            try:
                if str(json.loads(r['context']).get('channel',''))==ch:ids.append((r['id'],))
            except Exception:pass
        if ids:self.db.executemany('DELETE FROM experiences WHERE id=?',ids)
        self.db.commit()
    def close(self):
        try:self.db.close()
        except Exception:pass
