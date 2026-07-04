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
`token_embeddings`. `tokenizer`, `layers`, and `output` are optional. The only
supported tokenizer type today is `whitespace`.

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
