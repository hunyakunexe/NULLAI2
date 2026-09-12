import sqlite3,time,json,math
from pathlib import Path
class ExperienceCore:
 def __init__(self,path='data/experience.db',decay_days=60):
  Path(path).parent.mkdir(parents=True,exist_ok=True); self.decay_days=max(1,float(decay_days)); self.db=sqlite3.connect(path,check_same_thread=False); self.db.row_factory=sqlite3.Row; self.init()
 def init(self):
  self.db.executescript('''CREATE TABLE IF NOT EXISTS experiences(id INTEGER PRIMARY KEY,semantic TEXT,input TEXT,context TEXT,response TEXT,reactions TEXT,next_messages TEXT,result TEXT,score REAL,frequency INTEGER,confidence REAL,ts REAL,fingerprint TEXT);CREATE INDEX IF NOT EXISTS ex_sem ON experiences(semantic);CREATE INDEX IF NOT EXISTS ex_fp ON experiences(fingerprint);CREATE TABLE IF NOT EXISTS terms(term TEXT PRIMARY KEY,meaning TEXT,semantic TEXT,confidence REAL,source TEXT,ts REAL);CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,channel TEXT,guild TEXT,author TEXT,text TEXT,ts REAL);CREATE INDEX IF NOT EXISTS msg_channel ON messages(channel);CREATE TABLE IF NOT EXISTS profiles(kind TEXT,key TEXT PRIMARY KEY,stats TEXT,ts REAL);''');self.db.commit()
 def add_term(self,t,m,s='learned',c=.9,source='human'): self.db.execute('INSERT OR REPLACE INTO terms VALUES(?,?,?,?,?,?)',(t,m,s,c,source,time.time()));self.db.commit()
 def known(self): return {r['term'] for r in self.db.execute('SELECT term FROM terms')}
 def add_message(self,**x): self.db.execute('INSERT OR REPLACE INTO messages VALUES(?,?,?,?,?,?)',(x['id'],x.get('channel'),x.get('guild'),x.get('author'),x.get('text'),x.get('ts',time.time())));self.db.commit()
 def add_experience(self,**x):
  context=json.dumps(x.get('context',{}),ensure_ascii=False,sort_keys=True); fp=x.get('fingerprint','')
  row=self.db.execute('SELECT id,frequency FROM experiences WHERE fingerprint=? AND input=? AND response=? AND context=? ORDER BY id DESC LIMIT 1',(fp,x['input'],x['response'],context)).fetchone()
  vals=(x['semantic'],json.dumps(x.get('reactions',[]),ensure_ascii=False),json.dumps(x.get('next_messages',[]),ensure_ascii=False),x['result'],x['score'],x['confidence'],time.time())
  if row:self.db.execute('UPDATE experiences SET semantic=?,reactions=?,next_messages=?,result=?,score=?,frequency=?,confidence=?,ts=? WHERE id=?',(*vals,int(row['frequency'])+1,row['id']))
  else:self.db.execute('INSERT INTO experiences(semantic,input,context,response,reactions,next_messages,result,score,frequency,confidence,ts,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(x['semantic'],x['input'],context,x['response'],json.dumps(x.get('reactions',[]),ensure_ascii=False),json.dumps(x.get('next_messages',[]),ensure_ascii=False),x['result'],x['score'],1,x['confidence'],time.time(),fp))
  self.db.commit()
 def search(self,semantic,ctx,limit=16):
  rows=self.db.execute('SELECT * FROM experiences WHERE semantic=? ORDER BY ts DESC LIMIT ?', (semantic,max(1,limit*8))).fetchall();now=time.time();out=[]
  for r in rows:
   try:c=json.loads(r['context'])
   except Exception:c={}
   age=(now-r['ts'])/86400;decay=math.exp(-max(0,age)/self.decay_days);match=.2*(c.get('channel')==ctx.get('channel'))+.15*(c.get('guild')==ctx.get('guild'))+.1*(c.get('user')==ctx.get('user'));score=(r['score']*.55+r['confidence']*.2+min(1,r['frequency']/20)*.15+match)*decay;out.append((score,dict(r)))
  return [x[1] for x in sorted(out,key=lambda x:x[0],reverse=True)[:limit]]
 def export(self,path='data/curated_experiences.jsonl',min_score=.45,min_conf=.65,discord_path='data/discord.jsonl'):
  rows=self.db.execute('SELECT * FROM experiences WHERE score>=? AND confidence>=? ORDER BY ts ASC',(min_score,min_conf)).fetchall();Path(path).parent.mkdir(parents=True,exist_ok=True);Path(discord_path).parent.mkdir(parents=True,exist_ok=True)
  with open(path,'w',encoding='utf8') as f:
   for r in rows:f.write(json.dumps({'instruction':'Discord会話に自然に返信する','input':r['input'],'context':json.loads(r['context']),'output':r['response'],'result':r['result'],'score':r['score']},ensure_ascii=False)+'\n')
  with open(discord_path,'w',encoding='utf8') as f:
   for r in rows:f.write(json.dumps({'input':r['input'],'response':r['response'],'result':r['result'],'score':r['score'],'confidence':r['confidence'],'timestamp':r['ts']},ensure_ascii=False)+'\n')
  return len(rows)
 def forget_channel(self,ch):
  ch=str(ch); ids=[]
  for r in self.db.execute('SELECT id,context FROM experiences').fetchall():
   try:
    if str(json.loads(r['context']).get('channel',''))==ch: ids.append(r['id'])
   except Exception: pass
  if ids:self.db.executemany('DELETE FROM experiences WHERE id=?',[(i,) for i in ids])
  self.db.execute('DELETE FROM messages WHERE channel=?',(ch,));self.db.commit()
 def close(self):
  try:self.db.close()
  except Exception:pass
