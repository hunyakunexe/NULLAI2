# Discord Experience AI — Complete Final

仕様書の中心機能を一体化した実装です。

- 小型ローカルLLM接続点
- 意味グループ
- 経験グラフ(SQLite)
- 状況→AI返信→反応→結果の経験保存
- 成功度/頻度/信頼度/時間減衰
- サーバー/チャンネル/ユーザー文脈
- bot/DM/チャンネル収集制御
- 未知語を勝手に決めず質問
- `term = meaning` / `term は meaning` で人間から意味を学習
- 高品質経験の追加学習用JSONL出力API
- 忘却用コアAPI

## 起動
`.env` に `DISCORD_TOKEN`、任意でローカルモデルの `MODEL_PATH` を設定して `python main.py`。
`AUTO_REPLY` は安全のため初期OFF。config.jsonでONにする。

Discordの利用規約、サーバー管理者の許可、参加者へのデータ収集告知を確認して使用してください。

## 起動時の自動データ補充

Bot起動時に公式の青空文庫公開作品カタログを確認し、`data/pretrain.txt` が設定した最低量に届いていなければ、不足分だけ未取得作品を追加します。取得済み作品は `data/aozora/manifest.json` で記録されるため、次回起動では重複取得しません。

設定は `config.json` の `dataset` で変更できます。

- `min_corpus_chars`: 必要な最低本文文字数
- `max_new_works_per_start`: 1回の起動で追加する最大作品数
- `auto_fill`: 自動補充のON/OFF
- `training.auto_train_on_startup`: 新規データが追加された起動時に差分学習するか

Tokenizerは初回作成後に固定し、後から語彙を作り直さない構成です。


## 返信しないチャンネルの設定

管理者はDiscord上のコマンドで、AIが返信しないチャンネルを設定できます。設定は `config.json` に保存され、再起動後も維持されます。

- `/ep channel deny` — コマンドを実行したチャンネルを返信対象外にする
- `/ep channel deny <channel_id>` — 指定チャンネルを返信対象外にする
- `/ep channel allow` — コマンドを実行したチャンネルを返信対象に戻す
- `/ep channel allow <channel_id>` — 指定チャンネルを返信対象に戻す
- `/ep channel list` — 現在の返信対象外チャンネルを表示

返信対象外チャンネルでは、メンションされても返信しません。

## 完成した自律学習・推論機能
- SentencePiece(Unigram)によるサブワードTokenizerへ移行。初回学習時に`data/tokenizer.model`を生成し、以後固定。
- `train.py`をチャンク化＋DataLoaderでバッチ学習。`batch_size`と`gradient_accumulation_steps`を設定可能。
- ExperienceCoreは高品質な経験を`data/discord.jsonl`へ自動export。
- 起動時に経験データを再exportし、経験ファイルがモデルより新しければ自動差分学習。
- Discordで一定件数の成功/観測済み経験が蓄積されるとバックグラウンド学習を実行。
- `MiniLLM.generate()`は各Attention層のKV cacheを使用して逐次生成を高速化。
- 推論時は`inference.quantized=true`でLinear層を動的INT8量子化。
- 量子化は保存済みのFPモデルをロードしてから実行するため、学習用チェックポイントはFPのまま保持。


## v3 修正内容
- モデル/チェックポイントのバージョンを3へ更新し、旧モデルを自動再学習対象にしました。
- Discord経験データの重複をまとめ、frequencyを実際に加算するよう修正しました。
- 経験の減衰日数をconfig.jsonのdecay_daysに統一しました。
- チャンネル削除時に経験データも削除するよう修正しました。
- 推論を別スレッドへ逃がし、Discordイベントループをブロックしにくくしました。
- 起動場所に依存しないようmain.pyでプロジェクトルートへ移動します。
- mutable default、DB索引、初期化・チェックポイント周りなどの細かい不具合も修正しました。
- 既存のstartup_maintenance.pyは残し、初回学習を壊さない構成にしています。
