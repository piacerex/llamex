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

## 性能・互換性・メモリ効率の最適化

この段階では、次の観測・互換性・境界整備を完了済みの足場として扱う。

- [x] profile / benchmark の `top_components` と `top_layers` で次のボトルネック候補を確認できる。
- [x] GGUF 診断で未対応 architecture、tokenizer model、pre-tokenizer、RoPE variant、attention variant を明示できる。
- [x] 量子化 tensor は既存の F32 展開読み込みを保ったまま、compact payload schema として保持形式を選択できる。
- [x] chat template は実モデルの metadata 差分を fixture で確認しながら対応パターンを増やせる。
- [x] AtomVM / FPGA 向け backend は NxEXLA とは別の `Backend` 実装として、演算境界と fallback 状態を確認できる。

次の継続ステップでは、これらの足場を使って実モデルごとの対応範囲を広げる。
以下は本ロードマップ上で継続して進める未完了項目として扱う。

- [ ] compact tensor backend を追加し、Q4_0 の token embeddings / output weights
      以外の主要 weight を eager F32 展開せずに扱えるようにする。
- [ ] 既知未対応 architecture の runtime 実装を追加し、診断 blocker から
      supported path へ移せるモデルを増やす。
- [ ] 実機 FPGA runtime への delegation 境界を実装し、fallback 状態だけでなく
      FPGA backend 実行結果を検証できるようにする。

現在の観測基盤:

- `mix llamex.natural.smoke --json` は GGUF モデルでは `model_diagnostic` を含み、
  eager F32 expansion ratio や tensor payload summary を smoke 結果から確認できる。
  README でも natural smoke の GGUF diagnostic 出力を明示している。
- `compact_weight_estimate` で現在の eager F32 bytes、GGUF payload bytes、
  possible savings、expansion ratio をまとめて確認できる。
- generate profile、benchmark、natural smoke の GGUF JSON artifact で
  `compact_weight_estimate` を確認できることをテストで固定している。
- GGUF supported surface では Mistral / Qwen / Phi を known unsupported
  architecture として明示し、runtime 未実装 blocker を JSON / text 診断と
  load error で確認できる。
- tokenizer metadata surface では SentencePiece tokenizer model と Qwen2
  pre-tokenizer を known unsupported として明示し、対応済み tokenizer
  metadata と未対応候補を分けて確認できる。
- attention / RoPE variant surface では full attention / default RoPE と、
  sliding-window attention / linear・YaRN RoPE scaling の known unsupported
  status を分けて確認できる。
- `Llamex.GGUF.Reader.read_compact_tensor_data/2` で GGUF tensor payload を
  eager F32 展開せず named tensor schema として読めるため、量子化 tensor の
  メモリ効率の良い保持形式へ進む足場がある。
- `Llamex.GGUF.ModelLoader.to_model_map/3` は `tensor_format: :compact` で
  compact payload schema を返せるため、標準の dequantized 推論経路を保ったまま
  量子化 weight 保持形式を選択できる。
- `Llamex.TensorStore` は compact tensor payload を識別し、標準の dequantized
  model loader 経路では明示拒否するため、compact backend 境界が曖昧にならない。
- `Llamex.TensorStore.fetch_compact_tensor/2` で compact tensor の metadata と
  payload を raw model-map shape から分離して取得できる。
- `Llamex.TensorStore.dequantize_compact_tensor/1` で compact Q4_0 payload を
  必要時に F32 data へ展開でき、eager dequantized reader 経路と一致する。
- `Llamex.TensorStore.fetch_dequantized_matrix/2` で compact Q4_0 matrix を
  既存 backend が期待する matrix value として遅延取得できる。
- `Llamex.TensorStore.fetch_dequantized_token_embeddings/2` で compact Q4_0
  `token_embd.weight` から token id => embedding map を遅延構築できる。
- `Llamex.ModelLoader.from_compact_map/1` で compact model map から最小 `Model` を
  opt-in で構築でき、compact Q4_0 token embeddings と任意の `output.weight` を
  遅延展開できる。
- compact Q4_0 token embeddings から構築した最小 `Model` は List backend で
  1 token generation smoke を通せる。
- `Llamex.GGUF.ModelLoader.load/2` は `tensor_format: :compact` で GGUF file から
  compact Q4_0 embedding / output-weight path を opt-in でロードできる。
- chat template は ChatML / role marker / Llama header / Gemma turn marker に加え、
  Mistral・Llama2 系の `[INST]...[/INST]` marker を診断・適用できる。
- `Llamex.Backend.FPGA.capabilities/0` で FPGA backend の fallback 状態、
  dequantized tensor 境界、AtomVM-oriented 境界を確認できる。

## gemma3対応

現時点では、Gemma 3 の text-only / full attention / default RoPE 経路を
supported architecture として扱い、最小 GGUF fixture でロードと 1 token
生成までをテストで固定している。GGUF tokenizer は
`tokenizer.chat_template` と `tokenizer.ggml.chat_template` の両方を読み、
Gemma turn marker template を tokenizer に保持できる。最小 fixture では
`generate_chat/3` と `stream_chat/3` の 1 token 生成も通している。Gemma3
prefix の model config は embedding、context、layer、head、KV head、
feed-forward、RMSNorm epsilon、RoPE theta、RoPE dimension を `Model`
config へ変換している。Gemma3 tensor schema は post attention norm の
内部名変換に加え、output、attention、FFN tensor の shape を config から
診断でき、q/k extra norm と post feed-forward extra norm tensor を
GGUF load 後の model layer に保持できる。Gemma3 tokenizer fixture では special token、BOS 付与、
control token 除去、英語 encode、日本語 byte fallback decode を固定している。
`mix llamex.gguf.inspect --json` の Gemma3 fixture では architecture、
tokenizer model / pre-tokenizer、chat template、runtime capability、
model config、tensor schema の診断出力を固定している。Gemma3 loaded model
では `prefill/3` から `step/3` までの 1 token 生成と、List backend と
NxEXLA backend の greedy 1 token 生成一致を smoke test で確認している。
`mix llamex.natural.smoke` は `--include-japanese`
でデフォルト英語 prompt suite に日本語 prompt を追加でき、Gemma3 GGUF
fixture で英語 prompt と日本語 prompt の natural smoke を固定している。
README は Gemma3 text-only/full-attention/default-RoPE の supported surface、
未対応 feature blocker、inspect/load/generate/natural smoke の known-good
コマンドを記載している。`mix llamex.exla.info --target cuda|rocm --json`
は GPU target availability を機械判定できる smoke として固定している。
README の Gemma3 extra norm 説明は supported runtime と整合している。
README の tensor schema 診断説明も Gemma3 extra norm supported path と整合している。

完了証跡:

- `gguf inspect task can print supported surface without a model file` と
  `gguf inspect task can print supported surface as json without a model file`
  で Gemma3 supported surface を固定している。
- `diagnoses unsupported gemma3 attention and rope metadata by architecture prefix`
  で sliding-window attention と RoPE scaling の feature blocker を固定している。
- `gguf inspect task can print gemma3 json diagnostics` で Gemma3 metadata、
  tokenizer、chat template、runtime capability、model config、tensor schema を固定している。
- `loads gemma3 gguf models with supported text runtime variants` で load、
  `generate/3`、`prefill/3`、`step/3`、`generate_chat/3`、`stream_chat/3`、
  system / user / assistant role chat、List / NxEXLA 一致を固定している。
- `generate task profiles gemma3 gguf generation` で `mix llamex.generate`
  相当の CLI profile 経路を固定している。
- `loads gemma3 gguf extra norm tensors into model layers` で q/k extra norm と
  post feed-forward extra norm の loader 保持を固定している。
- `builds a gemma3 tokenizer with special tokens and byte fallback metadata` で
  special token、BOS、control token 除去、英語 encode、日本語 byte fallback decode を固定している。
- `natural smoke task runs English and Japanese prompts against a gemma3 gguf`
  で Gemma3 GGUF fixture の英語 / 日本語 natural smoke を固定している。
- `exla info task reports unavailable GPU targets` と
  `exla info task reports rocm target availability` で CUDA / ROCm availability JSON を固定している。

完了項目:

- [x] 対象モデルを `unsloth/gemma-3-270m-it-GGUF` の text-only 経路に固定する。
- [x] `mix llamex.gguf.inspect MODEL --json` 相当の Gemma3 診断差分を fixture で固定する。
- [x] GGUF 診断に Gemma3 supported surface と未対応 feature blocker を追加する。
- [x] Gemma3 metadata を architecture prefix から `Model` config へ変換する。
- [x] Gemma3 tensor 名を内部 schema へ対応させ、shape 検証を追加する。
- [x] Gemma3 tokenizer の special token、BOS / EOS、byte fallback、英語 / 日本語 decode を固定する。
- [x] Gemma3 の full attention / default RoPE / extra norm text runtime を supported path として固定する。
- [x] sliding-window attention と RoPE scaling は explicit feature blocker として固定する。
- [x] Gemma3 chat template の `system` / `user` / `assistant` role と `generate_chat/3` / `stream_chat/3` smoke を固定する。
- [x] `Llamex.GGUF.ModelLoader.load/1`、`prefill/3`、`step/3`、`mix llamex.generate`、`mix llamex.natural.smoke` の end-to-end smoke を固定する。
- [x] List backend を基準に NxEXLA の greedy token 選択一致を固定する。
- [x] CUDA / ROCm は `mix llamex.exla.info --target TARGET --json` で availability を機械判定できるように固定する。
- [x] README に supported GGUF surface、未対応 variant、known-good コマンドを記載する。
