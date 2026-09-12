"""起動時のデータ補充・経験export・必要な初期/差分学習を管理する。"""
import json
import torch
from pathlib import Path
from dataset_manager import update_dataset
from experience_core import ExperienceCore

MODEL_VERSION=3

def _checkpoint_compatible(ckpt,config,tok_path="data/tokenizer.model"):
    if not ckpt.exists() or not Path(tok_path).exists(): return False
    try:
        ck=torch.load(ckpt,map_location="cpu",weights_only=False)
        import sentencepiece as spm
        sp=spm.SentencePieceProcessor(model_file=tok_path)
        mc=ck.get("model_config",{})
        cfg=config
        return (ck.get("model_version")==MODEL_VERSION and ck.get("vocab_size")==sp.get_piece_size() and
                all(mc.get(k)==cfg[k] for k in ("n_embd","n_head","n_layer","block_size","dropout")))
    except Exception as exc:
        print(f"[startup] checkpoint check failed: {exc}",flush=True)
        return False

def run(config):
    result=update_dataset(config)
    core=ExperienceCore(decay_days=float(config.get("experience",{}).get("decay_days",60)))
    exported=core.export(min_score=float(config.get('experience',{}).get('export_score',.45)), min_conf=float(config.get('experience',{}).get('export_confidence',.65)))
    discord_path=Path('data/discord.jsonl'); ckpt=Path('data/model.pt')
    compatible=_checkpoint_compatible(ckpt,config)
    newer=discord_path.exists() and (not ckpt.exists() or discord_path.stat().st_mtime > ckpt.stat().st_mtime)
    print(f"[startup] pretrain corpus={result.get('chars',0):,} chars, added={result.get('added',0)}, failures={result.get('failures',0)}, curated_experiences={exported}, checkpoint_ok={compatible}",flush=True)
    training=config.get('training',{})
    needs_train=(not compatible) or result.get('added',0)>0 or newer
    if training.get('auto_train_on_startup',True) and needs_train:
        from train import train_once
        return train_once(incremental=compatible and ckpt.exists())
    return False

if __name__=='__main__':
    with open('config.json',encoding='utf8') as f: run(json.load(f))
