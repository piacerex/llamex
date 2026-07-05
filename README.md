# Llamex

Minimal Elixir LLM engine experiment.

## Run

```bash
mix test
mix llamex.generate priv/models/tiny.json hello 2
mix llamex.generate priv/models/tiny.json hello 2 --temperature 1.0 --top-k 1 --top-p 0.9 --seed 42
mix llamex.generate priv/models/tiny.json hello 2 --profile
mix llamex.generate model.gguf "The" 1 --natural --profile --candidates 5
mix llamex.generate priv/models/tiny.json hello 2 --profile --no-stop
mix llamex.generate priv/models/tiny.json hello 2 --stop-token 2
mix llamex.generate priv/models/tiny.json hello 2 --stop-piece world
mix llamex.generate model.gguf "Hello" 8 --stop-special eos
mix llamex.generate model.gguf "Hello" 8 --natural --stop-control --profile
mix llamex.generate priv/models/tiny.json hello 2 --backend nx
mix llamex.generate model.gguf "Hello" 8 --natural
mix llamex.tokenize model.gguf "Elixir is"
```

## Backends

Llamex keeps the core path on `Llamex.Backend.List` so the engine remains
portable to restricted runtimes such as AtomVM. Nx is available as an optional
dependency for BEAM experiments:

```elixir
state = Llamex.prefill(model, "hello", %{backend: Llamex.Backend.Nx})
step = Llamex.step(state.context, state.current_token, %{sampler: :greedy})
```

EXLA can be added by BEAM-only consumers that want an XLA compiler for Nx:

```elixir
{:exla, "~> 0.12.0"}
```

The Nx backend prepares projection matrices as Nx tensors when a context is
created, but selecting it alone does not yet make existing GGUF generation fast.
The next speed step is reducing Nx preparation overhead and moving more of the
matvec-heavy layer execution onto Nx/EXLA while keeping the List backend as the
AtomVM reference path.

## Model JSON

Llamex can load a small JSON model with:

```elixir
Llamex.ModelLoader.load_json("priv/models/tiny.json")
```

The current schema is intentionally small:

```json
{
  "config": {
    "vocab_size": 3,
    "embedding_size": 2,
    "context_size": 128,
    "epsilon": 1.0e-6,
    "rope_theta": 10000.0
  },
  "tokenizer": {
    "type": "whitespace",
    "unknown_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "hello": 1,
      "world": 2
    }
  },
  "token_embeddings": {
    "0": [0.0, 0.0],
    "1": [1.0, 0.0],
    "2": [2.0, 0.0]
  },
  "layers": [],
  "output": {
    "weight": [[1.0, 0.0]]
  }
}
```

Required fields are `config.vocab_size`, `config.embedding_size`, and
either `token_embeddings` or `tensors.token_embd.weight`. `tokenizer`, `layers`,
`output_norm`, and `output` are optional.

Named tensors use this shape:

```json
{
  "tensors": {
    "token_embd.weight": {
      "shape": [3, 2],
      "dtype": "f32",
      "data": [0.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    }
  }
}
```

The current tensor reader validates `shape`, `dtype`, and flat `data`, then maps
`token_embd.weight` into Llamex token embeddings. Supported dtypes are `f32` and
`f16`; values are still represented as Elixir numbers after loading.

Recognized transformer tensor names:

- `token_embd.weight`
- `blk.N.attn_norm.weight`
- `blk.N.attn_q.weight`
- `blk.N.attn_k.weight`
- `blk.N.attn_v.weight`
- `blk.N.attn_output.weight`
- `blk.N.ffn_norm.weight`
- `blk.N.ffn_gate.weight`
- `blk.N.ffn_up.weight`
- `blk.N.ffn_down.weight`
- `output_norm.weight`
- `output.weight`

Supported tokenizer types:

- `whitespace`: splits on whitespace and looks up whole tokens.
- `bpe`: applies a small ordered merge list to each whitespace-delimited word.

The BPE implementation is fixture-oriented and not byte-level yet. See
`priv/models/tiny_bpe.json` for the current shape.
GGUF byte tokens such as `<0x68>` are encoded as a fallback and decoded from
`tokenizer.ggml.token_type` metadata when present.

Tokenizer JSON files with a BPE `model` can also be loaded directly:

```elixir
Llamex.Tokenizer.Loader.load_tokenizer_json("priv/tokenizers/tiny_tokenizer.json")
```

Model JSON can reference such a file with:

```json
{
  "tokenizer": {
    "path": "priv/tokenizers/tiny_tokenizer.json"
  }
}
```

## GGUF

Llamex can read GGUF header, metadata, and tensor directory information:

```elixir
Llamex.GGUF.Reader.read_metadata("model.gguf")
```

The metadata reader validates the `GGUF` magic, reads v3-style header counts,
metadata values, tensor names, tensor dimensions, tensor types, and tensor data
offsets.

F32, F16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1, and Q8_K tensor data can be read into Llamex's named tensor schema:

```elixir
Llamex.GGUF.Reader.read_tensors("model.gguf")
```

Rank-2 GGUF tensor dimensions are normalized into Llamex schema order when
building the JSON-style tensor map. Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1, and Q8_K tensors are
dequantized to F32 values while loading. Other quantized tensor types are not
loaded yet.

Small GGUF files can be loaded as Llamex models:

```elixir
Llamex.GGUF.ModelLoader.load("model.gguf")
```

The `mix llamex.generate` task accepts `.gguf` paths and uses the GGUF loader
for them. Use `--natural` to select a conservative text sampling preset
(`temperature=0.8`, `top-k=40`, `top-p=0.9`, `repetition-penalty=1.1`).
Use `--profile` to inspect the model path, prompt, prompt token IDs/pieces,
generation settings, generated token IDs/pieces/types, timings, and
`finish_reason` (`stop` or `length`) for generation experiments. Profile output
also splits prefill into `prompt_encode`, `backend_prepare`, and `prompt_eval`
timings so backend setup cost is visible. Each generated step includes
`eval_timings` with per-layer `attention_norm`, `attention`, `mlp`,
`output_norm`, and `logits` timings. For List backend top-k sampling, the
profile labels the shortened output projection as `top_k_logits`. `mlp` is
further split into `feed_forward_norm`, `w_gate_up`, `silu_multiply`, `w_down`,
and `residual` on the List backend. Add `--candidates N` with `--profile` to
inspect the top sampled candidate token pieces and probabilities for each
generated step.
Use `--stop-token ID`, `--stop-piece TOKEN`, `--stop-special eos`, or
`--stop-control` to override inferred EOS/stop behavior, or `--no-stop` to force
generation to continue until `max_new_tokens`.
Use `mix llamex.tokenize` to inspect prompt token IDs, pieces, and GGUF token
types before choosing stop pieces or chat prompts.
Use `--chat` only after `mix llamex.gguf.inspect` reports that the chat template
has no missing tokens. The generate task validates `.gguf --chat` from metadata
before loading tensors, so incompatible chat templates fail quickly.

Current GGUF generation baseline on
`zephyr-smol_llama-100m-sft-full-Q2_K.gguf` with the List backend:

```bash
mix llamex.generate /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf "The" 3 --natural --stop-control
mix llamex.generate /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf "Once upon a time" 3 --natural --stop-control
```

Verified multi-token runs now reach ordinary text generation. The prompt `The`
generated `Yarmed`, while `Once upon a time` generated `of high stress`. The
shorter prompt is still low quality, but the longer prompt shows the existing
GGUF path can produce natural word pieces instead of decode noise. On the
current development machine this remains slow: one-token profiled runs spend
tens of seconds in prefill and about 11s in the sampled step. Profile timings
show the next speed targets are the feed-forward `w_down` matvecs and the final
`logits` matvec.

GGUF compatibility can be inspected without loading tensor data:

```bash
mix llamex.gguf.inspect model.gguf
mix llamex.gguf.inspect model.gguf --json
mix llamex.gguf.inspect first.gguf second.gguf --json
```

The inspection output includes special tokens, chat template support, and
missing marker tokens, plus representative raw GGUF dimensions and normalized
schema shapes for key tensors. This is the fastest way to decide whether
`--chat` is safe for a checkpoint and whether tensor layout looks plausible.
Use `chat_usable: true` in JSON output as the quick check for `--chat` readiness.
With `--json`, multiple GGUF paths can be inspected in one command for model
candidate comparison.

An existing tiny GGUF model can be smoke-tested without checking the model file
into the repository:

```bash
mkdir -p /tmp/llamex-models
curl -L --fail \
  -o /tmp/llamex-models/test-gguf-trainer.Q8_0.gguf \
  https://huggingface.co/ybelkada/test-gguf-trainer-Q8_0-GGUF/resolve/main/test-gguf-trainer.Q8_0.gguf

mix llamex.gguf.inspect /tmp/llamex-models/test-gguf-trainer.Q8_0.gguf
mix run -e 'model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/test-gguf-trainer.Q8_0.gguf"); IO.inspect(%{vocab_size: model.config.vocab_size, embedding_size: model.config.embedding_size, layers: length(model.layers), token_embeddings: map_size(model.token_embeddings)})'
```

For a small existing GGUF that reaches the generation path, use the
`tensorblock/zephyr-smol_llama-100m-sft-full-GGUF` Q2_K file:

```bash
mkdir -p /tmp/llamex-models
curl -L --fail \
  -o /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf \
  https://huggingface.co/tensorblock/zephyr-smol_llama-100m-sft-full-GGUF/resolve/main/zephyr-smol_llama-100m-sft-full-Q2_K.gguf

mix llamex.gguf.inspect /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf
```

This checkpoint is compatible with the current tensor loader:

```text
architecture: llama
tokenizer tokens: 32128
supported tensor types: F32=13, Q2_K=25, Q3_K=18, Q6_K=1
unsupported tensor types: none
```

It can also run a sampled generation step:

```bash
mix run -e 'model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf"); profile = Llamex.Profile.generation_steps(model, "Elixir is", %{backend: Llamex.Backend.List, max_new_tokens: 1, sampler: %{temperature: 0.8, top_k: 40, top_p: 0.9, repetition_penalty: 1.1, seed: 1}}); IO.inspect(profile, limit: :infinity)'
```

On the current List backend this is still slow for natural prose. Treat the
commands above as the current known-good existing GGUF path; the remaining work
for natural multi-sentence text is backend speed and tokenizer/template quality.

For iterative GGUF testing in IEx, load the model once and step tokens without
reloading the file:

```elixir
model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/test-gguf-trainer.Q8_0.gguf")
state = Llamex.prefill(model, "hello", %{backend: Llamex.Backend.List})
step = Llamex.step(state.context, state.current_token, %{sampler: :greedy})
step.text
```

The same loaded model can be profiled for one prefill and generation step:

```elixir
Llamex.Profile.generation_step(model, "hello", %{backend: Llamex.Backend.List})
```

Tokenizer metadata can be converted into a Llamex tokenizer:

```elixir
gguf = Llamex.GGUF.Reader.read_metadata("model.gguf")
tokenizer = Llamex.GGUF.Tokenizer.from_metadata(gguf.metadata)
```

Currently recognized tokenizer keys:

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.merges`
- `tokenizer.ggml.unknown_token_id`
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.padding_token_id`
- `tokenizer.ggml.add_bos_token`
- `tokenizer.ggml.add_eos_token`
- `tokenizer.ggml.token_type`

Currently mapped Llama config keys:

- `llama.vocab_size`
- `llama.embedding_length`
- `llama.context_length`
- `llama.block_count`
- `llama.attention.head_count`
- `llama.attention.head_count_kv`
- `llama.feed_forward_length`
- `llama.attention.layer_norm_rms_epsilon`
- `llama.rope.freq_base`

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `llamex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llamex, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/llamex>.
