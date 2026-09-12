# DiscordExperienceAI v8

v7の社会記憶を維持したまま、生成品質・学習安定性・経験検索・起動時互換性を修正した版。

## 主な修正
- incremental学習でSentencePiece tokenizerを再構築しない
- 既存の互換checkpointを再利用し、差分学習では全量の事前学習を繰り返さない
- gradient accumulationの最終stepも必ず更新
- 経験検索に入力文の類似度を追加し、無関係な経験を回答へ流用しにくくした
- 直接返信/メンションを優先した回答評価
- 生成プロンプトの話者ラベルを明示
- prompt/control markerの漏出を除去
- 文字・n-gramの無限反復を検出・抑制
- fallbackを自然な短文へ変更
- 社会記憶DBをWALで安定化
- 社会記憶に保持期限を追加
- モデルversionを8へ更新

## 学習方針
変化の速いユーザー関係・活動度・サーバー傾向はDBに保持し、返信時に参照する。安定した良質な応答例だけをモデルの追加学習へ回す。
