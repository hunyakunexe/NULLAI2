import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
try:
 from dotenv import load_dotenv
 load_dotenv(ROOT/".env")
except Exception: pass
from discord_bot import Bot
with open(ROOT/"config.json",encoding="utf-8") as f: cfg=json.load(f)
t=os.getenv("DISCORD_TOKEN")
if not t: raise SystemExit("DISCORD_TOKEN is required (set it in the environment or .env)")
Bot(cfg).run(t)
