# Llamex

Minimal Elixir LLM engine experiment.

## Run

```bash
mix test
mix llamex.generate priv/models/tiny.json hello 2
mix llamex.generate priv/models/tiny.json hello 2 --stream --no-stop
mix llamex.generate priv/models/tiny.json hello 2 --temperature 1.0 --top-k 1 --top-p 0.9 --min-p 0.05 --seed 42
mix llamex.generate priv/models/tiny.json hello 2 --profile
mix llamex.generate model.gguf "The" 1 --natural --profile --candidates 5
mix llamex.generate priv/models/tiny.json hello 2 --profile --no-stop
mix llamex.generate priv/models/tiny.json hello 2 --stop-token 2
mix llamex.generate priv/models/tiny.json hello 2 --stop-piece world
mix llamex.generate model.gguf "Hello" 8 --stop-special eos
mix llamex.generate model.gguf "Hello" 8 --natural --stop-control --profile
mix llamex.generate model.gguf "Long prompt..." 32 --context-window 2048
mix llamex.generate priv/models/tiny.json hello 2 --backend list
mix llamex.generate priv/models/tiny.json hello 2 --backend nx_exla
mix llamex.generate priv/models/tiny.json hello 2 --backend nx_exla --exla cpu
mix llamex.generate priv/models/tiny.json hello 2 --backend nx_exla --exla cuda
mix llamex.generate priv/models/tiny.json hello 2 --backend nx_exla --exla rocm
mix llamex.generate priv/models/tiny.json hello 2 --backend fpga
mix llamex.generate model.gguf "Hello" 8 --natural
mix llamex.tokenize model.gguf "Elixir is"
mix llamex.natural.baseline model.gguf --json
mix llamex.benchmark model.gguf --tokens 8,16,24,32 --backend nx_exla --exla cpu --natural --json
```

## Backends

Llamex keeps `Llamex.Backend.List` as the pure Elixir reference path so the
engine remains portable to restricted runtimes such as AtomVM. PC-oriented
commands default to the Nx backend, and backend selection is explicit:

- `Llamex.Backend.Nx`: default Nx path, backed by `Llamex.Backend.NxEXLA`
- `Llamex.Backend.List`: pure Elixir reference path
- `Llamex.Backend.NxEXLA`: optional Nx/EXLA path for BEAM experiments
- `Llamex.Backend.FPGA`: FPGA boundary, currently backed by the List fallback

Greedy and top-k sampled generation are regression-tested to agree across List,
Nx, and NxEXLA on the same tiny model when Nx is available.

```text
PC:   operation wrapper -> Llamex.Backend.NxEXLA
FPGA: operation wrapper -> Llamex.Backend.FPGA
```

Nx is available as an optional dependency for BEAM experiments:

```elixir
state = Llamex.prefill(model, "hello", %{backend: Llamex.Backend.NxEXLA})
step = Llamex.step(state.context, state.current_token, %{sampler: :greedy})
```

EXLA can be added by BEAM-only consumers that want an XLA compiler for Nx:

```elixir
{:exla, "~> 0.12.0"}
```

EXLA targets can be selected from the CLI:

```bash
mix llamex.exla.info --target cpu
mix llamex.exla.info --target cuda --json
mix llamex.exla.info --target rocm --json
mix llamex.generate model.gguf "Hello" 8 --backend nx_exla --exla cpu
mix llamex.generate model.gguf "Hello" 8 --backend nx_exla --exla cuda
mix llamex.generate model.gguf "Hello" 8 --backend nx_exla --exla rocm
```

`--exla cpu` maps to EXLA's `:host` client. `--exla cuda` and `--exla rocm`
require matching XLA binaries and GPU runtime setup, including `XLA_TARGET`.
Use `mix llamex.exla.info --target TARGET --json` and check
`target_available?` before running a GPU target.
The same configuration is available from Elixir with
`Llamex.Backend.NxEXLA.configure!(:cpu | :cuda | :rocm)`.
Generation profile JSON includes the configured EXLA target under `exla`.

The NxEXLA backend prepares projection matrices as Nx tensors when a context is
created, but selecting it alone does not yet make existing GGUF generation fast.
The next speed step is reducing Nx preparation overhead and moving more of the
matvec-heavy layer execution onto Nx/EXLA while keeping the List backend as the
AtomVM reference path.

`--context-window` keeps the tail of long prompts and caps generation so the
minimal KV cache does not step past the selected window.

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

Gemma 3 diagnostics also recognize architecture-specific tensor names before
runtime support is enabled. `blk.N.post_attention_norm.weight` is mapped to the
internal `blk.N.ffn_norm.weight` schema for model-map inspection, while
`blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight`, and
`blk.N.post_ffw_norm.weight` are reported as unsupported tensor features until
the runtime implements those extra norms.

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

Current GGUF load support is intentionally narrow:

- Architecture: `llama`
- Tokenizer kinds: `whitespace`, `bpe`
- Tokenizer model metadata: `llama`, `gpt2`, or omitted
- Pre-tokenizer metadata: `default`, `gpt2`, `llama-bpe`, or omitted
- Tensor types: `F32`, `F16`, `BF16`, `Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`,
  `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_1`, `Q8_K`

GGUF files for other runtime architectures such as Mistral, Qwen, Gemma, or Phi
are not loadable yet. Gemma 3 is a known diagnostic architecture: `gemma3.*`
model metadata is inspected with the Gemma 3 prefix, so Gemma 3 checkpoints can
show model config and tensor diagnostics before the architecture runtime exists.
The GGUF model-map conversion path can also read `gemma3.*` config metadata, but
`Llamex.GGUF.ModelLoader.load/1` still rejects Gemma 3 with an unsupported
architecture runtime issue until the runtime is implemented. Llama checkpoints
that require sliding-window attention or non-`none` RoPE scaling are also
rejected by the compatibility check.

F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1, and Q8_K tensor data can be read into Llamex's named tensor schema:

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
(`temperature=0.8`, `top-k=40`, `top-p=0.5`, `repetition-penalty=1.1`,
`no-repeat-ngram-size=2`, `no-repeat-adjacent-word=true`).
Use `--min-p VALUE` to keep tokens whose probability is at least
`VALUE * max_probability` after temperature/top-k filtering. `top-k` must be a
positive integer, and `top-p`, `min-p`, and `repetition-penalty` are validated
before sampling. `seed` must be a non-negative integer, and a fixed `random`
override must be a float in `[0.0, 1.0)`.
Use `--profile` to inspect the model path, prompt, prompt token IDs/pieces,
generation settings, generated token IDs/pieces/types, timings, and
`finish_reason` (`stop` or `length`) for generation experiments. Profile output
also includes `model_diagnostic` for GGUF models, with the same eager F32
payload expansion, chat usability, and tokenizer metadata issue fields used by
benchmark JSON. It splits prefill into `prompt_encode`,
`backend_prepare`, and `prompt_eval` timings so backend setup cost is visible. Each generated step includes
`eval_timings` with per-layer `attention_norm`, `attention`, `mlp`,
`output_norm`, and `logits` timings. For List backend top-k sampling, the
profile labels the shortened output projection as `top_k_logits`. `mlp` is
further split into `feed_forward_norm`, `w_gate_up`, `silu_multiply`, `w_down`,
and `residual` on the List backend. Add `--candidates N` with `--profile` to
inspect the top sampled candidate token pieces and probabilities for each
generated step. Natural-mode profile, generation, and stream results report
`suppressed_token_count` instead of printing the full internal suppression list. Use
`Llamex.Natural.suppressed_token_ids(model)` in IEx to inspect the exact token
IDs suppressed by the natural preset.
Use `--stream` to write generated token text as chunks are produced.
Use `--stop-token ID`, `--stop-piece TOKEN`, `--stop-special eos`, or
`--stop-control` to override inferred EOS/stop behavior, or `--no-stop` to force
generation to continue until `max_new_tokens`. Use `--stop-sequence TEXT` to
stop when decoded generated text contains a string sequence.
Use `mix llamex.tokenize` to inspect prompt token IDs, pieces, and GGUF token
types before choosing stop pieces or chat prompts.
Use `--chat` only after `mix llamex.gguf.inspect` reports that the chat template
has no missing tokens. The generate task validates `.gguf --chat` from metadata
before loading tensors, so incompatible chat templates fail quickly.
Use `mix llamex.natural.smoke MODEL [max_new_tokens] --json` to run the current
natural-generation baseline prompts after loading the model once. Smoke results
include `ok` and `issues` fields for raw `▁` markers, suppressed token types, or
punctuation-only output, and adjacent repeated words.
Add `--min-words N` to require generated text to contain at least that many word
fragments.
Add `--reject-open-ending` to report length-limited output that ends on an
alphanumeric fragment or non-terminal punctuation such as a comma.
Add `--complete-open-ending N` to let the smoke task generate up to `N` extra
tokens, in small chunks, while that open ending is detected.
Add `--trim-to-sentence` to keep the last complete sentence and report any
discarded trailing text in JSON output.
Add `--fail-on-issue` to make the task raise when any prompt reports issues.
Use `mix llamex.natural.baseline MODEL --json` for the current stricter GGUF
baseline gate. It defaults to the known-good `The quick brown fox` prompt with
8 initial tokens, at least 4 generated word fragments, incomplete-ending rejection,
up to 8 completion tokens, sentence trimming, and fail-on-issue enabled.
JSON smoke output includes the `model_path`, prompt/generated/completion token
IDs and pieces, and `settings` used for each prompt, including the backend and
EXLA target metadata, so baseline results are auditable.

Use `mix llamex.benchmark MODEL --tokens 8,16,24,32 --json` to compare
generation cost across multiple output lengths after loading the model once.
The benchmark task reports raw warmup and measured runs plus summary values
for total, prefill, step, eval, per-generated-token milliseconds, and
tokens-per-second. It prepares the model once per backend before warmup and
measured runs, reporting that one-time cost as `backend_prepare_milliseconds`
in JSON and `backend_prepare_ms` in text output. Each JSON run includes prompt
and generated token pieces, so benchmark output can be matched back to tokenizer
text fragments without a separate profile run. For GGUF models, JSON results
also include `model_diagnostic` with payload expansion fields such as
`gguf_payload_bytes`, `eager_f32_expansion_ratio`, `tensor_payload_by_type`, and
`top_tensor_payloads`, plus chat usability and tokenizer metadata issue fields,
so speed, eager F32 memory cost, and checkpoint metadata quality can be compared
in the same artifact. When comparing backends, each
result also includes `comparison_rank`, `comparison_fastest_backend`, and
`mean_milliseconds_delta_from_fastest` for the same requested token count. It
accepts backend comparison, EXLA, natural sampler, context window, stop-control,
sampling, and `--trim-to-sentence` options used by the generation and smoke
tasks. For example:

```bash
mix llamex.benchmark /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf \
  --tokens 8,16,24,32 \
  --backends list,nx_exla \
  --exla cpu \
  --natural \
  --context-window 96 \
  --stop-control \
  --trim-to-sentence \
  --warmup 1 \
  --repeat 3 \
  --json
```

When using EXLA, keep at least one warmup run because the first execution can
include compiler and cache setup. Use measured-run `summary.mean`,
`summary.median`, and `summary.best` values, not a single cold run, when
deciding whether List or NxEXLA is faster for a prompt and token count.
Each raw benchmark run also includes `prompt_eval_steps` and
`prompt_eval_summary`. JSON benchmark runs include `prepared?`,
prompt/generated token IDs/pieces, context-window limits, prompt truncation
metadata, and the sampler settings, so the prepared route, generation budget,
seed, and sampling options used for each run can be audited. Use
`prompt_eval_steps` to inspect prefill token-by-token timings, and
`prompt_eval_summary.layers` to compare accumulated prefill layer costs across
List and NxEXLA. Non-JSON benchmark output also prints
`prompt_eval_top_layers` and `prompt_eval_top_components` so the next prefill
optimization target is visible without expanding the raw JSON.

Current GGUF generation baseline on
`zephyr-smol_llama-100m-sft-full-Q2_K.gguf` with the List backend:

```bash
mix llamex.natural.baseline /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf --json
```

The current verified baseline output for the default prompt is:

```text
. They were a bit of brown and brown.
```

This is not high-quality prose yet, but it is ordinary decoded text from an
existing GGUF path with `ok: true` and no smoke issues after trimming the
incomplete trailing fragment. Short prompts can still produce weak
continuations, but the existing GGUF path can produce natural word pieces
instead of decode noise. Byte-token output is normalized through the same
SentencePiece-style decoder, so standalone `▁` markers are not leaked into
generated text. The `--natural` preset also suppresses byte tokens,
unknown/unused tokens, non-EOS control tokens, standalone `▁`,
newline/carriage-return pieces, repeated token n-grams, and adjacent repeated
words while sampling. On the current development machine this remains slow:
one-token profiled runs spend tens of seconds in prefill and about 11s in the
sampled step. Profile timings show the next speed targets are the feed-forward
`w_down` matvecs and remaining layer matvec work.

For quick manual comparisons, shorter direct generation still works:

```bash
mix llamex.generate /tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf "The quick brown fox" 3 --natural --stop-control
```

### IEx Backend Examples

Use these snippets for quick backend comparisons against the same loaded GGUF
model and prompt.

#### Default Nx Backend

```elixir
prompt = "The quick brown fox"
model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")

start_time = DateTime.utc_now()

result =
  Llamex.generate(model, prompt, %{
    backend: Llamex.Backend.Nx,
    max_new_tokens: 8,
    stop_tokens: Llamex.Natural.control_stop_tokens(model),
    sampler: Llamex.Natural.sampler(model)
  })

IO.inspect DateTime.diff(DateTime.utc_now(), start_time, :second) / 60

result.sampler
result.prepared?
result.text
```

#### Nx EXLA CPU

```elixir
Llamex.Backend.NxEXLA.configure!(:cpu)

prompt = "The quick brown fox"
model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")

start_time = DateTime.utc_now()

result =
  Llamex.generate(model, prompt, %{
    backend: Llamex.Backend.NxEXLA,
    max_new_tokens: 8,
    stop_tokens: Llamex.Natural.control_stop_tokens(model),
    sampler: Llamex.Natural.sampler(model)
  })

IO.inspect DateTime.diff(DateTime.utc_now(), start_time, :second) / 60

result.text
```

#### Prepared Nx EXLA Reuse

Prepare once when reusing the same GGUF model for multiple prompts. This keeps
backend tensor preparation outside each generation call.

```elixir
Llamex.Backend.NxEXLA.configure!(:cpu)

model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")
prepared = Llamex.prepare_model(model, Llamex.Backend.NxEXLA)

prompt = "The quick brown fox"
start_time = DateTime.utc_now()

result =
  Llamex.generate(prepared, prompt, %{
    max_new_tokens: 8,
    stop_tokens: Llamex.Natural.control_stop_tokens(prepared),
    sampler: Llamex.Natural.sampler(prepared)
  })

IO.inspect DateTime.diff(DateTime.utc_now(), start_time, :second) / 60

result.text
```

#### Token Streaming

Use `stream/3` to receive token chunks as they are generated. Each chunk
includes `:token`, `:text`, `:prompt_tokens`, `:prompt_pieces`,
`:generated_tokens`, `:generated_pieces`, `:sampler`, `:prepared?`, `:context`,
and `:finish_reason`; the final length-limited chunk has `token: nil`.

```elixir
prepared
|> Llamex.stream("The quick brown fox", %{
  max_new_tokens: 8,
  stop_sequences: ["</s>"],
  sampler: Llamex.Natural.sampler(prepared)
})
|> Enum.each(fn chunk ->
  IO.write(chunk.text)
end)
```

#### Nx EXLA CUDA GPU

```elixir
Llamex.Backend.NxEXLA.configure!(:cuda)

prompt = "The quick brown fox"
model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")

start_time = DateTime.utc_now()

result =
  Llamex.generate(model, prompt, %{
    backend: Llamex.Backend.NxEXLA,
    max_new_tokens: 8,
    stop_tokens: Llamex.Natural.control_stop_tokens(model),
    sampler: Llamex.Natural.sampler(model)
  })

IO.inspect DateTime.diff(DateTime.utc_now(), start_time, :second) / 60

result.text
```

#### Nx EXLA ROCm GPU

```elixir
Llamex.Backend.NxEXLA.configure!(:rocm)

prompt = "The quick brown fox"
model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")

start_time = DateTime.utc_now()

result =
  Llamex.generate(model, prompt, %{
    backend: Llamex.Backend.NxEXLA,
    max_new_tokens: 8,
    stop_tokens: Llamex.Natural.control_stop_tokens(model),
    sampler: Llamex.Natural.sampler(model)
  })

IO.inspect DateTime.diff(DateTime.utc_now(), start_time, :second) / 60

result.text
```

Generation and stream results include `prompt_pieces` and `generated_pieces` so
token IDs can be matched back to tokenizer text fragments without running a
separate profile command.

#### Prepared Chat Generation

Use `generate_chat/3` when the tokenizer has a supported chat template. It
formats the prompt or message list first, then runs normal generation. Supported
roles are `system`, `user`, and `assistant`; role-marker templates that do not
have a separate system marker fold system messages into the user marker, while
templates with `<|system|>` use it for system messages. Each message must be a
map with string-compatible `role` and string `content`.
When template markers are missing from the tokenizer, inspect the model with
`mix llamex.gguf.inspect MODEL_GGUF` before using chat generation.

```elixir
Llamex.Backend.NxEXLA.configure!(:cpu)

model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")
prepared = Llamex.prepare_model(model, Llamex.Backend.NxEXLA)

messages = [
  %{role: "system", content: "Be concise."},
  %{role: "user", content: "Explain Elixir processes in one sentence."}
]

result =
  Llamex.generate_chat(prepared, messages, %{
    max_new_tokens: 32,
    stop_tokens: Llamex.Natural.control_stop_tokens(prepared),
    sampler: Llamex.Natural.sampler(prepared)
  })

result.text
```

String stop sequences are also available from the public API:

```elixir
Llamex.generate(prepared, prompt, %{
  max_new_tokens: 32,
  stop_sequences: ["</s>", "\nUser:"],
  sampler: Llamex.Natural.sampler(prepared)
})
```

Empty stop sequences are ignored; non-string stop sequences raise
`ArgumentError` so generation settings fail fast.
Stop tokens must be non-negative integer token IDs for the same reason.
`max_new_tokens` must also be a non-negative integer.
When set, `context_window` must be a positive integer.

Use `stream_chat/3` with a supported chat template when streaming chat prompts.

#### List Backend

```elixir
prompt = "The quick brown fox"
model = Llamex.GGUF.ModelLoader.load("/tmp/llamex-models/zephyr-smol_llama-100m-sft-full-Q2_K.gguf")

start_time = DateTime.utc_now()

result =
  Llamex.generate(model, prompt, %{
    backend: Llamex.Backend.List,
    max_new_tokens: 8,
    stop_tokens: Llamex.Natural.control_stop_tokens(model),
    sampler: Llamex.Natural.sampler(model)
  })

IO.inspect DateTime.diff(DateTime.utc_now(), start_time, :second) / 60

result.text
```

GGUF compatibility can be inspected without loading tensor data:

```bash
mix llamex.gguf.inspect --supported
mix llamex.gguf.inspect --supported --json
mix llamex.gguf.inspect model.gguf
mix llamex.gguf.inspect model.gguf --json
mix llamex.gguf.inspect model.gguf --config
mix llamex.gguf.inspect first.gguf second.gguf --config --json
mix llamex.gguf.inspect model.gguf --schema
mix llamex.gguf.inspect first.gguf second.gguf --schema --json
mix llamex.gguf.inspect first.gguf second.gguf --json
```

The tensor schema summary can also be checked from IEx without reading tensor
payloads:

```elixir
Llamex.GGUF.ModelLoader.model_config_report_file("model.gguf")
Llamex.GGUF.ModelLoader.model_config_summary_file("model.gguf")
Llamex.GGUF.ModelLoader.tensor_schema_summary_file("model.gguf")
```

`model_config_report_file/1` includes the selected metadata prefix, such as
`llama` or `gemma3`, alongside the config map and missing config metadata keys.
`Llamex.GGUF.ModelLoader.model_config_summary/1` and
`Llamex.GGUF.ModelLoader.tensor_schema_summary/1` accept an already parsed
`Llamex.GGUF.Reader` when caller code wants to reuse metadata.

The inspection output includes supported architecture/tokenizer/tensor type
combinations, architecture runtime status, special tokens, tokenizer `add_bos` /
`add_eos` flags, chat template support/family, missing marker tokens, tokenizer model
support/kind/merge counts, score counts, tokenizer metadata issues, token type
counts, mapped Llama model config, missing model config metadata,
unsupported attention/RoPE feature metadata, plus representative raw GGUF
dimensions and normalized schema shapes for key tensors. This is the fastest way
to decide whether `--chat` is safe for a checkpoint and whether tensor layout
looks plausible.
Per-model diagnostics include the tokenizer metadata surface selected for that
checkpoint's architecture, so unsupported tokenizer model or pre-tokenizer
values can be compared with the accepted values directly.
Use `--supported` without a model path to print the current supported GGUF
surface, architecture runtime status, model config metadata mapping, supported
tokenizer metadata mapping, tensor type ID/name pairs, and explicitly
unsupported feature metadata before choosing a checkpoint. It also
includes the tensor schema surface for known architectures so Gemma 3 extra norm
tensor names are visible before loading tensor data.
`known_combinations` includes diagnostic-only architectures such as Gemma 3 with
their runtime status, while `supported_combinations` remains limited to loadable
runtime combinations.
Use `chat_usable: true` in JSON output as the quick check for `--chat` readiness.
Use `chat_template_issues: []` to confirm that the template is supported and all
required marker tokens are present.
`loadable: true` and `chat_usable: true` are separate checks: a model can be
loadable for plain prompt generation while still lacking a usable chat template.
Use `supported_tensors` and `unsupported_tensors` in JSON output to see the
per-tensor type, dimensions, and quantization mix behind the aggregate tensor
type counts.
Compare `gguf_payload_bytes` with `eager_f32_bytes` or
`eager_f32_expansion_ratio` to estimate the memory cost of the current eager F32
expansion path before moving a checkpoint to a more compact quantized
representation.
Use `tensor_payload_by_type` to find which tensor types contribute most to that
expansion before choosing the next compact in-memory representation.
Use `top_tensor_payloads` to identify the largest individual tensors on the
current eager F32 path.
Use `tensor_schema_mappings` to see architecture-specific GGUF tensor names
that are normalized into Llamex's internal tensor schema, such as Gemma 3
post-attention norms. Use `tensor_schema_issues: []` to confirm that no tensor
names remain outside the current schema mapping, and
`unsupported_tensor_features: []` to confirm that no recognized tensor names
still require unsupported runtime behavior such as Gemma 3 extra norms.
Supported chat templates currently cover ChatML, `<|user|>`/`<|assistant|>`
role markers, and Llama header markers using `<|start_header_id|>`,
`<|end_header_id|>`, and `<|eot_id|>`, including templates that start with
`<|begin_of_text|>`. Gemma turn markers using `<start_of_turn>` and
`<end_of_turn>` are also supported; system messages are folded into the first
user turn for that format. `tokenizer_metadata_issues` reports Gemma/chat marker
tokens that are present but not marked as control tokens.
Use `loadable: true` as the quick check that architecture, tokenizer metadata,
tokenizer model metadata when present, required model metadata, required tensors,
required tensor schema names, required tensor shapes, and tensor types are
inside Llamex's current supported GGUF surface.
Use `compatibility_issues: []` in JSON output to confirm that no unsupported
architecture, tokenizer metadata, tokenizer model, required model metadata, or
required tensor, tensor schema mapping issue, tensor shape mismatch, or tensor
type was found.
When `unsupported_features` is non-empty, inspect
`unsupported_feature_metadata_values` to see the exact GGUF metadata values such
as sliding-window size or RoPE scaling settings that caused the rejection. These
feature checks follow the model architecture prefix, for example `llama.*` or
`gemma3.*`.
`Llamex.GGUF.ModelLoader.load/1` uses the same compatibility checks before
loading tensor data.
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
supported architectures: llama
supported combinations: llama+whitespace/bpe+llama/gpt2+default/gpt2/llama-bpe+BF16/F16/F32/Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K
architecture supported: true
supported tokenizers: whitespace, bpe
tokenizer supported: true
supported tokenizer models: llama, gpt2
tokenizer model supported: true
supported pre-tokenizers: default, gpt2, llama-bpe
pre-tokenizer supported: true
model config: ...
loadable: true
compatibility issues: none
tokenizer model: unknown
pre-tokenizer: unknown
tokenizer kind: whitespace
tokenizer tokens: 32128
tokenizer merges: 0
supported tensor type names: BF16, F16, F32, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1, Q8_K
supported tensor types: F32=13, Q2_K=25, Q3_K=18, Q6_K=1
eager f32 lower bound: ...
gguf payload bytes: ...
eager f32 expansion ratio: ...
tensor payload by type: ...
top tensor payloads:
- ...
supported tensors:
- token_embd.weight: Q2_K [256, 32128]
- ...
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

The result includes prompt token IDs/pieces, sampled token info/text, prefill
timings, and step timings.
Use `Llamex.Profile.prefill_steps/3` when you only need prompt-eval timings; it
reports backend, EXLA target metadata, prepared status, prompt token IDs/pieces,
and per-token prefill timings.
Profile JSON includes `timing_summary.top_components` and
`timing_summary.top_layers` sorted by elapsed milliseconds. Use those fields to
pick the next backend optimization target from the measured run instead of
guessing from the model structure.

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
