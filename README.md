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
and `output` are optional.

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

Supported tokenizer types:

- `whitespace`: splits on whitespace and looks up whole tokens.
- `bpe`: applies a small ordered merge list to each whitespace-delimited word.

The BPE implementation is fixture-oriented and not byte-level yet. See
`priv/models/tiny_bpe.json` for the current shape.

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
