# Llamex

Minimal Elixir LLM engine experiment.

## Run

```bash
mix test
mix llamex.generate priv/models/tiny.json hello 2
mix llamex.generate priv/models/tiny.json hello 2 --temperature 1.0 --top-k 1 --top-p 0.9 --seed 42
```

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

F32, F16, Q2_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1, and Q8_K tensor data can be read into Llamex's named tensor schema:

```elixir
Llamex.GGUF.Reader.read_tensors("model.gguf")
```

Rank-2 GGUF tensor dimensions are normalized into Llamex schema order when
building the JSON-style tensor map. Q2_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1, and Q8_K tensors are
dequantized to F32 values while loading. Other quantized tensor types are not
loaded yet.

Small F32 GGUF files can be loaded as Llamex models:

```elixir
Llamex.GGUF.ModelLoader.load("model.gguf")
```

The `mix llamex.generate` task accepts `.gguf` paths and uses the GGUF loader
for them.

GGUF compatibility can be inspected without loading tensor data:

```bash
mix llamex.gguf.inspect model.gguf
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
