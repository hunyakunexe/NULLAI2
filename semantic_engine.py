import re,hashlib
RULES=[('greeting',r'^(おはよ|こんにちは|こんばんは|やあ|おつ)'),('game_invite',r'(ゲーム|マイクラ|minecraft|遊ぶ|やる).*(する|やろ|やる|行く)'),('question',r'(？|\?)$|^(なん|なに|どう|いつ|どこ|誰|なぜ|なんで)'),('agreement',r'^(うん|はい|そう|それな|わかる|了解|いいよ|OK|おけ)'),('disagreement',r'(違う|いや|無理|ダメ|そうじゃない)'),('joke',r'(w+|草|笑|www|ネタ|冗談)')]
def norm(s):return re.sub(r'\s+',' ',s.lower().strip())
def analyze(s,known=None):
 known=set() if known is None else known
 t=norm(s); sem='general_chat'
 for n,p in RULES:
  if re.search(p,t,re.I):sem=n;break
 words=re.findall(r'[一-龯々ぁ-んァ-ヶA-Za-z0-9_]+',t); unknown=[w for w in words if w not in known and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{2,}',w) and w.lower() not in {'minecraft','discord','python','llm','ai','bot'}]
 return {'semantic':sem,'unknown':unknown,'fingerprint':hashlib.sha256(t.encode()).hexdigest()[:24]}
