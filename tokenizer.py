"""SentencePiece tokenizer used by the local LLM."""
from pathlib import Path
import json

class SentencePieceTokenizer:
    def __init__(self, model_path="data/tokenizer.model"):
        import sentencepiece as spm
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.itos = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def bos_id(self):
        return int(self.sp.bos_id())

    @property
    def eos_id(self):
        return int(self.sp.eos_id())

    @property
    def pad_id(self):
        return int(self.sp.pad_id())

    def encode(self, text, add_bos=True, add_eos=True):
        """Encode text. Generation prompts must not end in EOS."""
        ids = self.sp.encode(str(text), out_type=int)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        # Decode only content tokens; EOS/BOS/PAD are structural tokens.
        clean = [int(i) for i in ids if int(i) not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(clean)

    def save_info(self, path="data/tokenizer.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type":"sentencepiece", "model":self.model_path,
                       "vocab_size":self.vocab_size}, f, ensure_ascii=False, indent=2)

    @classmethod
    def train(cls, texts, model_path="data/tokenizer.model", vocab_size=8000):
        import sentencepiece as spm
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        corpus = Path(model_path).with_suffix(".corpus.txt")
        with corpus.open("w", encoding="utf-8") as f:
            for t in texts:
                t = str(t).strip()
                if t:
                    f.write(t.replace("\x00", " ") + "\n")
        prefix = str(Path(model_path).with_suffix(""))
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=prefix,
            vocab_size=int(vocab_size), model_type="unigram",
            character_coverage=0.9995, bos_id=1, eos_id=2, unk_id=3, pad_id=0,
            hard_vocab_limit=False, user_defined_symbols=[]
        )
        corpus.unlink(missing_ok=True)
        return cls(model_path)

def load_tokenizer(path="data/tokenizer.model"):
    return SentencePieceTokenizer(path)
