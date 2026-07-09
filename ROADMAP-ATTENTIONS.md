# ROADMAP-ATTENTIONS.md

## 目的

Gemma3 270m GGUF のような sliding-window attention / GQA 系モデルで、
Attention 実装を差し替えることによる性能改善余地を検証する。

このロードマップでは、まず既存の List / NxEXLA backend 上で効果を測り、
効果が確認できたものを `Llamex.Layers.Attention` と `Backend` 境界へ段階的に取り込む。

## 現状

- `Llamex.Layers.Attention.forward/8` は token ごとに Q/K/V を計算する。
- KV cache は保持している。
- `attention_sliding_window` がある場合は、参照 entries を直近 N 件に制限する。
- GQA / MQA では `attention_head_count` と `attention_head_count_kv` を使い、
  query head から KV head を選択する。
- FlashAttention、block sparse attention、paged KV cache、backend-native tensor cache、
  attention 専用 fused kernel は未実装。

## 改善候補

- [ ] sliding-window 専用 attention を追加する。
  - window 外の KV entry を早い段階で除外する。
  - KV cache 側に window pruning / ring buffer 境界を追加できるか検証する。
  - 現在の `Enum.take(entries, window)` より前段で保持量を抑えられるか確認する。

- [ ] GQA / MQA 専用 attention を追加する。
  - `attention_head_count > attention_head_count_kv` のとき、KV head 再利用を明示する。
  - query head ごとの重複処理を減らせるか確認する。
  - Gemma3 270m の `attention_head_count=4` / `attention_head_count_kv=1` を基準に測る。

- [ ] Q/K/V projection 融合を進める。
  - 既存の `Backend.qkv_heads/8` 境界を Gemma3 の実 shape に合わせて活用する。
  - List backend では正しさ確認を優先し、NxEXLA backend で効果を測る。
  - Q/K/V matvec を個別実行する場合との差分を profile で比較する。

- [ ] NxEXLA 向け attention 実装を分離する。
  - List backend の単純実装とは別に、backend-native tensor を維持する経路を検討する。
  - 小さい token 数ではオーバーヘッドが勝つ可能性があるため、
    prompt 長 / `max_new_tokens` 別に測定する。

- [ ] KV cache 表現を見直す。
  - 現在の扱いやすい list 表現から、window ring buffer または backend-native cache へ進める。
  - sliding-window モデルでは長文時の保持量と prepare cost を重点的に測る。
  - 既存の `Llamex.KVCache` API を壊さずに差し替え可能か確認する。

- [ ] output logits / FFN との支配率を比較する。
  - Attention だけを最適化しても、全体では output projection や FFN が支配的な可能性がある。
  - `mix llamex.generate --profile` と benchmark JSON で `attention`、`qkv`、
    `ffn`、`output logits` の比率を確認する。

## 検証順

1. Gemma3 270m Q5_K_M GGUF で baseline profile を取得する。
2. List backend で sliding-window pruning の保持量と結果一致を固定する。
3. GQA 専用経路を小さな fixture でテストし、List backend で結果一致を固定する。
4. NxEXLA backend で Q/K/V projection 融合の効果を測る。
5. KV cache 表現を差し替え、長い prompt / 複数 token 生成で比較する。
6. 改善が確認できた経路だけを標準 Attention 経路へ統合する。

## 完了条件

- [ ] Gemma3 270m GGUF の baseline profile が保存されている。
- [ ] sliding-window attention の最適化前後で生成結果一致を確認できる。
- [ ] GQA / MQA 専用経路の fixture と実モデル smoke が通る。
- [ ] List / NxEXLA の少なくとも一方で、Attention 関連時間の改善を数値で確認できる。
- [ ] README または ROADMAP に、採用した Attention 経路と制約が反映されている。
