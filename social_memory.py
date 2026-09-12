"""Discord内の社会的記憶。観測可能な統計だけを保存し、性格を断定しない。"""
import json,math,re,sqlite3,time
from pathlib import Path

class SocialMemory:
    def __init__(self,path='data/social_memory.db',decay_days=45,retention_days=180):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=NORMAL')
        self.decay_days=max(1,float(decay_days)); self.retention_days=max(30,float(retention_days)); self._init()
    def _init(self):
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS user_profile(guild_id TEXT NOT NULL,user_id TEXT NOT NULL,display_name TEXT DEFAULT '',messages INTEGER DEFAULT 0,chars INTEGER DEFAULT 0,questions INTEGER DEFAULT 0,replies INTEGER DEFAULT 0,mentions_received INTEGER DEFAULT 0,unique_contacts INTEGER DEFAULT 0,last_seen REAL DEFAULT 0,hour_hist TEXT DEFAULT '{}',topic_hist TEXT DEFAULT '{}',style_hist TEXT DEFAULT '{}',influence_score REAL DEFAULT 0,PRIMARY KEY(guild_id,user_id));
        CREATE TABLE IF NOT EXISTS relation(guild_id TEXT NOT NULL,src_user TEXT NOT NULL,dst_user TEXT NOT NULL,interactions INTEGER DEFAULT 0,replies INTEGER DEFAULT 0,mentions INTEGER DEFAULT 0,positive INTEGER DEFAULT 0,negative INTEGER DEFAULT 0,last_seen REAL DEFAULT 0,PRIMARY KEY(guild_id,src_user,dst_user));
        CREATE TABLE IF NOT EXISTS server_profile(guild_id TEXT PRIMARY KEY,messages INTEGER DEFAULT 0,chars INTEGER DEFAULT 0,active_users INTEGER DEFAULT 0,last_seen REAL DEFAULT 0,hour_hist TEXT DEFAULT '{}',topic_hist TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS channel_profile(guild_id TEXT NOT NULL,channel_id TEXT NOT NULL,messages INTEGER DEFAULT 0,chars INTEGER DEFAULT 0,last_seen REAL DEFAULT 0,topic_hist TEXT DEFAULT '{}',PRIMARY KEY(guild_id,channel_id));
        CREATE INDEX IF NOT EXISTS rel_src ON relation(guild_id,src_user); CREATE INDEX IF NOT EXISTS rel_dst ON relation(guild_id,dst_user);
        ''');self.db.commit()
    @staticmethod
    def _load(s):
        try:return json.loads(s or '{}')
        except Exception:return {}
    @staticmethod
    def _dump(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
    @staticmethod
    def _inc(d,key,n=1,limit=80):
        k=str(key);d[k]=int(d.get(k,0))+int(n)
        if len(d)>limit:
            for k,_ in sorted(d.items(),key=lambda kv:kv[1])[:len(d)-limit]:d.pop(k,None)
    @staticmethod
    def _style(text):
        t=str(text).strip()
        if not t:return 'empty'
        if '?' in t or '？' in t:return 'question'
        if re.search(r'[!！]{1,}|[wｗ]{2,}|草$',t):return 'casual'
        if len(t)<=12:return 'short'
        if len(t)>=100:return 'long'
        return 'normal'
    def _edge(self,guild,src,dst,replies=0,mentions=0,positive=0,negative=0,ts=None):
        ts=float(ts or time.time());guild,src,dst=map(str,(guild,src,dst))
        if not guild or not src or not dst or src==dst:return
        row=self.db.execute('SELECT 1 FROM relation WHERE guild_id=? AND src_user=? AND dst_user=?',(guild,src,dst)).fetchone()
        if row:self.db.execute('UPDATE relation SET interactions=interactions+1,replies=replies+?,mentions=mentions+?,positive=positive+?,negative=negative+?,last_seen=? WHERE guild_id=? AND src_user=? AND dst_user=?',(int(replies),int(mentions),int(positive),int(negative),ts,guild,src,dst))
        else:self.db.execute('INSERT INTO relation VALUES(?,?,?,?,?,?,?,?,?)',(guild,src,dst,1,int(replies),int(mentions),int(positive),int(negative),ts))
    def record_message(self,*,guild_id,channel_id,user_id,display_name,text,concept='その他',mentioned_user_ids=None,replied_user_id=None,ts=None,bot_user_id=None):
        guild_id,channel_id,user_id=str(guild_id or ''),str(channel_id),str(user_id)
        if not guild_id:return
        ts=float(ts or time.time()); text=str(text or '').strip(); mentioned=list(dict.fromkeys(str(x) for x in (mentioned_user_ids or []) if str(x)!=user_id)); replied=str(replied_user_id) if replied_user_id and str(replied_user_id)!=user_id else None
        row=self.db.execute('SELECT * FROM user_profile WHERE guild_id=? AND user_id=?',(guild_id,user_id)).fetchone()
        if row:
            h,t,s=self._load(row['hour_hist']),self._load(row['topic_hist']),self._load(row['style_hist']);self._inc(h,time.localtime(ts).tm_hour);self._inc(t,concept);self._inc(s,self._style(text))
            self.db.execute('UPDATE user_profile SET display_name=?,messages=messages+1,chars=chars+?,questions=questions+?,replies=replies+?,mentions_received=mentions_received+?,last_seen=?,hour_hist=?,topic_hist=?,style_hist=? WHERE guild_id=? AND user_id=?',(str(display_name or ''),len(text),int('?' in text or '？' in text),int(replied is not None),len(mentioned),ts,self._dump(h),self._dump(t),self._dump(s),guild_id,user_id))
        else:self.db.execute('INSERT INTO user_profile(guild_id,user_id,display_name,messages,chars,questions,replies,mentions_received,last_seen,hour_hist,topic_hist,style_hist) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(guild_id,user_id,str(display_name or ''),1,len(text),int('?' in text or '？' in text),int(replied is not None),len(mentioned),ts,self._dump({str(time.localtime(ts).tm_hour):1}),self._dump({concept:1}),self._dump({self._style(text):1})))
        sr=self.db.execute('SELECT * FROM server_profile WHERE guild_id=?',(guild_id,)).fetchone()
        if sr:
            h,t=self._load(sr['hour_hist']),self._load(sr['topic_hist']);self._inc(h,time.localtime(ts).tm_hour);self._inc(t,concept);self.db.execute('UPDATE server_profile SET messages=messages+1,chars=chars+?,last_seen=?,hour_hist=?,topic_hist=? WHERE guild_id=?',(len(text),ts,self._dump(h),self._dump(t),guild_id))
        else:self.db.execute('INSERT INTO server_profile VALUES(?,?,?,?,?,?,?)',(guild_id,1,len(text),1,ts,self._dump({str(time.localtime(ts).tm_hour):1}),self._dump({concept:1})))
        cr=self.db.execute('SELECT * FROM channel_profile WHERE guild_id=? AND channel_id=?',(guild_id,channel_id)).fetchone()
        if cr:
            t=self._load(cr['topic_hist']);self._inc(t,concept);self.db.execute('UPDATE channel_profile SET messages=messages+1,chars=chars+?,last_seen=?,topic_hist=? WHERE guild_id=? AND channel_id=?',(len(text),ts,self._dump(t),guild_id,channel_id))
        else:self.db.execute('INSERT INTO channel_profile VALUES(?,?,?,?,?,?)',(guild_id,channel_id,1,len(text),ts,self._dump({concept:1})))
        targets=set(mentioned); 
        if replied:targets.add(replied)
        for dst in targets:
            self._edge(guild_id,user_id,dst,replies=int(dst==replied),mentions=int(dst in mentioned),ts=ts)
        contacts=self.db.execute('SELECT COUNT(DISTINCT dst_user) FROM relation WHERE guild_id=? AND src_user=?',(guild_id,user_id)).fetchone()[0]
        self.db.execute('UPDATE user_profile SET unique_contacts=? WHERE guild_id=? AND user_id=?',(contacts,guild_id,user_id))
        active=self.db.execute('SELECT COUNT(*) FROM user_profile WHERE guild_id=? AND last_seen>?',(guild_id,ts-30*86400)).fetchone()[0]
        self.db.execute('UPDATE server_profile SET active_users=? WHERE guild_id=?',(active,guild_id));self._recompute_influence(guild_id,user_id);self.db.commit()
    def _recompute_influence(self,guild_id,user_id):
        row=self.db.execute('SELECT * FROM user_profile WHERE guild_id=? AND user_id=?',(str(guild_id),str(user_id))).fetchone()
        if not row:return
        indeg=self.db.execute('SELECT COALESCE(SUM(mentions+replies),0) n,COUNT(DISTINCT src_user) c FROM relation WHERE guild_id=? AND dst_user=?',(str(guild_id),str(user_id))).fetchone()
        score=.45*math.log1p(row['messages'])+.15*math.log1p(row['unique_contacts'])+.25*math.log1p(indeg['n'])+.15*math.log1p(indeg['c'])
        self.db.execute('UPDATE user_profile SET influence_score=? WHERE guild_id=? AND user_id=?',(score,str(guild_id),str(user_id)))
    def observe_bot_result(self,guild_id,user_id,bot_user_id,positive=False,negative=False,ts=None):
        guild_id,user_id,bot_user_id=str(guild_id or ''),str(user_id),str(bot_user_id)
        if not guild_id or user_id==bot_user_id:return
        self._edge(guild_id,user_id,bot_user_id,positive=int(positive),negative=int(negative),ts=ts);self._edge(guild_id,bot_user_id,user_id,positive=int(positive),negative=int(negative),ts=ts);self.db.commit()
    def context_for(self,guild_id,user_id,bot_user_id=None,recent_user_ids=None,channel_id=None):
        guild_id,user_id=str(guild_id or ''),str(user_id)
        if not guild_id:return {}
        row=self.db.execute('SELECT * FROM user_profile WHERE guild_id=? AND user_id=?',(guild_id,user_id)).fetchone();server=self.db.execute('SELECT * FROM server_profile WHERE guild_id=?',(guild_id,)).fetchone()
        def top(hist,n=3):return [k for k,v in sorted(self._load(hist).items(),key=lambda x:x[1],reverse=True)[:n]] if hist else []
        out={'server':{},'user':{},'relationships':[]}
        if server:out['server']={'messages':server['messages'],'active_users':server['active_users'],'active_hours':top(server['hour_hist'],3),'topics':top(server['topic_hist'],5)}
        if row:out['user']={'messages':row['messages'],'avg_chars':round(row['chars']/max(1,row['messages']),1),'question_rate':round(row['questions']/max(1,row['messages']),2),'reply_rate':round(row['replies']/max(1,row['messages']),2),'influence_score':round(row['influence_score'],2),'style':top(row['style_hist'],3),'topics':top(row['topic_hist'],5),'active_hours':top(row['hour_hist'],3)}
        recent=list(dict.fromkeys(str(x) for x in (recent_user_ids or []) if str(x)!=user_id))[:6]
        for uid in recent:
            r=self.db.execute('SELECT r.*,u.display_name FROM relation r LEFT JOIN user_profile u ON u.guild_id=r.guild_id AND u.user_id=r.dst_user WHERE r.guild_id=? AND r.src_user=? AND r.dst_user=?',(guild_id,user_id,uid)).fetchone()
            if r:out['relationships'].append({'name':r['display_name'] or '相手','interactions':r['interactions'],'replies':r['replies'],'mentions':r['mentions']})
        return out
    @staticmethod
    def prompt_text(ctx):
        if not ctx:return ''
        s=['社会コンテキスト（参考統計。人格の断定には使わない）:'];u=ctx.get('user',{})
        if u:
            if u.get('style'):s.append('話し方の傾向: '+', '.join(u['style']))
            if u.get('topics'):s.append('最近の話題: '+', '.join(u['topics']))
        sv=ctx.get('server',{})
        if sv and sv.get('topics'):s.append('サーバーの話題: '+', '.join(sv['topics']))
        rel=ctx.get('relationships',[])
        if rel:s.append('直近の相手との関係: '+', '.join(f"{x['name']}({x['interactions']})" for x in rel[:3]))
        return '\n'.join(s)
    def cleanup(self):
        cutoff=time.time()-self.retention_days*86400
        self.db.execute('DELETE FROM relation WHERE last_seen<?',(cutoff,));self.db.execute('DELETE FROM user_profile WHERE last_seen<?',(cutoff,));self.db.execute('DELETE FROM channel_profile WHERE last_seen<?',(cutoff,));self.db.commit()
    def close(self):
        try:self.db.close()
        except Exception:pass
