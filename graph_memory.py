import sqlite3, time, re, hashlib
from pathlib import Path
from difflib import SequenceMatcher

class GraphMemory:
    """
    発言を大きな意味カテゴリにまとめ、
    状態 + 条件 -> 次の発言 の遷移を記録する簡易会話グラフ。
    """
    def __init__(self, path="data/memory.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS utterance(
            id INTEGER PRIMARY KEY, user_id TEXT, text TEXT, concept TEXT,
            created REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS transition(
            src_concept TEXT, condition TEXT, dst_text TEXT, count INTEGER,
            PRIMARY KEY(src_concept, condition, dst_text))""")
        self.db.commit()

    def concept(self, text):
        t = re.sub(r"\s+", "", text.lower())
        groups = [
            ("挨拶", ["こんにちは","こんばんは","おはよう","やあ","どうも"]),
            ("空腹", ["腹減","お腹す","腹へ","食いたい","食べたい"]),
            ("疲労", ["疲れ","眠い","だるい","しんどい"]),
            ("ゲーム", ["ゲーム","マイクラ","minecraft","遊ぶ"]),
            ("質問", ["？","?","どうして","なぜ","何","どうやって"]),
            ("感謝", ["ありがとう","感謝","助かった"]),
            ("開発", ["コード","プログラム","bot","python","バグ","開発"]),
        ]
        for name, words in groups:
            if any(w in t for w in words):
                return name
        # 未知の発言は正規化したハッシュで安定した概念IDにする
        return "未知:" + hashlib.sha1(t.encode()).hexdigest()[:10]

    def add(self, user_id, text, previous_text=None, condition="default"):
        c = self.concept(text)
        self.db.execute("INSERT INTO utterance(user_id,text,concept,created) VALUES(?,?,?,?)",
                        (str(user_id), text, c, time.time()))
        if previous_text:
            src = self.concept(previous_text)
            self.db.execute("""INSERT INTO transition(src_concept,condition,dst_text,count)
                VALUES(?,?,?,1) ON CONFLICT(src_concept,condition,dst_text)
                DO UPDATE SET count=count+1""", (src, condition, text))
        self.db.commit()

    def next_candidates(self, previous_text, condition="default", limit=5):
        src = self.concept(previous_text)
        return self.db.execute("""SELECT dst_text,count FROM transition
            WHERE src_concept=? AND condition=? ORDER BY count DESC LIMIT ?""",
            (src, condition, limit)).fetchall()
