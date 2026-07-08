# ROADMAP

## 方針

直近の最適化で、NxEXLA 実行、KV cache、prepared model、benchmark / profile の基盤が整ってきた。
次の段階では、手製の最小モデルを速く動かすだけでなく、実在する Llama 系モデルを自然文で扱える LLM 実装へ進める。

## 完遂後の到達像

このロードマップを完遂した状態では、`llamex` は実在する Llama 系 GGUF モデルを読み込み、自然文 prompt から token を生成できる最小 LLM 推論ランタイムになる。

- tokenizer により、文字列 prompt を token id に変換できる。
- GGUF の metadata、tensor、tokenizer 情報を `Model` に取り込める。
- prepared model を使って、prompt eval から token sampling、decode まで実行できる。
- temperature、top-k、top-p、repeat penalty などで生成の振る舞いを調整できる。
- chat template により、instruct / chat model 向けの prompt を組み立てられる。
- benchmark / profile を使い、実モデル上の性能を測定しながら NxEXLA 最適化を進められる。

## 継続的な品質・性能基盤

各機能追加では、既存の高速化を壊さないことを前提にする。

- List / Nx / NxEXLA backend の結果一致を回帰テストで固定する。
- prepared model、KV cache、RoPE、attention の最適化パスを仕様として扱う。
- benchmark / profile の代表ケースを決め、変更前後の比較に使う。
- 次の最適化対象は推測ではなく profile 結果から選ぶ。
- 性能改善は public API や backend 責務境界を崩さない範囲で行う。

## 現在の進捗

現時点では、当面の完了目標に必要な主要経路は一通り実装済みと扱う。

- tokenizer による自然文 encode / decode。
- GGUF metadata、tokenizer、tensor の読み込み。
- F32 / F16 / BF16 と主要 Q 系 tensor の F32 展開読み込み。
- prepared model、KV cache、RoPE、attention を通した生成経路。
- List / Nx / NxEXLA backend の差し替え。
- temperature、top-k、top-p、min-p、repeat penalty、seed 指定。
- stop token、stop sequence、streaming、chat template。
- benchmark / profile による backend 比較と計測。

以降は、実モデル上での性能・互換性・メモリ効率を伸ばす段階とする。
変更は profile 結果、互換性診断、smoke test のいずれかで効果を確認できる
単位に分ける。

## 優先ロードマップ

### 1. トークナイザ対応

- 自然文入力から token id 列を作れるようにする。
- SentencePiece、BPE、GGUF 内 tokenizer metadata の扱い方を決める。
- special token、BOS / EOS、未知語、byte fallback の扱いを明確にする。
- 日本語 prompt を encode / decode できる smoke test を追加する。

### 2. GGUF 読み込み

- GGUF の metadata と tensor info を読み込む。
- まずは F32 / F16 / BF16 相当の非量子化 weight を対象にする。
- tokenizer 情報、モデル構成、tensor 名の対応を `Model` の責務へ統合する。

### 3. 実モデルの end-to-end 推論

- Llama 系の小規模 GGUF モデルを読み込み、自然文 prompt から logits / token 生成まで通す。
- List / Nx / NxEXLA backend の結果差分を確認できるテストを追加する。
- prepared model を標準的な推論経路に組み込む。
- 日本語 prompt を使った end-to-end smoke test を追加する。

### 4. Sampler 強化

- greedy 以外の sampling を追加する。
- temperature、top-k、top-p、min-p、repeat penalty を段階的に実装する。
- 再現性確認のため seed 指定の扱いも設計する。

### 5. 生成 API の実用化

- 文字列 prompt を受け取り、文字列または token stream を返す API を整える。
- max tokens、stop token、stop sequence、streaming を扱う。
- prepared model reuse を自然に使える public API にする。

### 6. チャットテンプレート対応

- instruct model 向けに chat template を適用できるようにする。
- GGUF metadata に含まれる template を優先して利用する。
- system / user / assistant role と special token の対応を明確にする。

### 7. 量子化 weight 対応

- GGUF の量子化 tensor を扱う。
- 最初の対象候補は Q8_0 または Q4_0 とする。
- 量子化の展開処理は backend の責務境界と整合させる。

## 対応モデルの考え方

最初の対象は、GGUF 化された Llama 系アーキテクチャの小規模モデルとする。

- まずは Llama 2 / Llama 3 / Llama 3.1 / Llama 3.2 系の 1B、3B、7B、8B クラスを優先する。
- 最初の weight 形式は F32 / F16 / BF16 相当、量子化では Q8_0 または Q4_0 を候補にする。
- Mistral 系は、GQA や sliding window attention などの差分を確認しながら次の候補にする。
- Qwen 系、Gemma 系、Phi 系は、tokenizer、RoPE variant、attention、chat template、metadata の差分を吸収してから広げる。
- GGUF であれば常に動くとは扱わず、対応済みの architecture、tokenizer、quantization type の組み合わせを明示する。

## 当面の完了目標

最初の大きな節目は、次の流れを一つの経路で動かすこととする。

```text
GGUF モデル読み込み
-> tokenizer による自然文 encode
-> prepared model 作成
-> prompt eval
-> sampler による token 選択
-> tokenizer による decode
```

この節目に到達した後、benchmark / profile を使って NxEXLA の次のボトルネックを選び、実モデル上の性能改善へ進む。

## 節目到達状況

当面の完了目標は、現在の `llamex` の主要経路として到達済みと扱う。

- `.gguf` は `Llamex.GGUF.ModelLoader.load/1` で読み込み、互換性診断を通ったモデルだけを `Model` に変換する。
- tokenizer metadata は `Llamex.GGUF.Tokenizer.from_metadata/1` から `Model` の tokenizer に統合する。
- prompt は `Llamex.encode/2` と `Llamex.prefill/3` で token id 列と推論状態に変換する。
- prepared model は `Llamex.prepare_model/2` と `Llamex.generate/3` / `Llamex.stream/3` で再利用できる。
- sampler は greedy、temperature、top-k、top-p、min-p、repeat penalty、seed、抑制 token を扱う。
- decode は `Llamex.decode/2`、`generate/3`、`stream/3`、`generate_chat/3`、`stream_chat/3` で実行できる。
- `mix llamex.natural.smoke` と `mix llamex.natural.baseline` で実モデル向けの自然文 smoke を実行できる。
- `mix llamex.gguf.inspect` で architecture、tokenizer、tensor type、model config、chat template、互換性 issue を tensor load 前に確認できる。
- `mix llamex.benchmark` と `Llamex.Profile` で backend 比較、prompt eval、top timing、context window の影響を測定できる。

この状態は、`mix test` の全テスト成功と、README に記載した既存 GGUF smoke / baseline / benchmark の手順で検証する。

## 次フェーズ

以降は「機能を通す」段階ではなく、実モデル上の性能・互換性・メモリ効率を伸ばす段階とする。

- profile / benchmark の `top_components` と `top_layers` を見て NxEXLA の次のボトルネックを選ぶ。
- GGUF 診断で未対応 architecture、tokenizer model、pre-tokenizer、RoPE variant、attention variant を明示し、対応範囲を広げる。
- 量子化 tensor は既存の F32 展開読み込みを基準に、メモリ効率の良い保持形式へ進める。
- chat template は実モデルの metadata 差分を smoke で確認しながら対応パターンを増やす。
- AtomVM / FPGA 向け backend は NxEXLA とは別の `Backend` 実装として、演算境界を保ったまま検証する。
