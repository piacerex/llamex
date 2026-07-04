defmodule LlamexTest do
  use ExUnit.Case

  test "runs one minimal greedy inference step" do
    model =
      Llamex.new_model(%{
        config: %{vocab_size: 3, embedding_size: 2},
        token_embeddings: %{
          0 => [1.0, 0.0],
          1 => [0.0, 1.0],
          2 => [0.8, 0.2]
        }
      })

    context = Llamex.new_context(model, Llamex.Backend.List)

    {context, next_token} = Llamex.next_token(context, 0)

    assert context.tokens == [0]
    assert next_token == 0
  end

  test "encodes text with the minimal tokenizer" do
    tokenizer = Llamex.Tokenizer.new(%{"<unk>" => 0, "hello" => 1, "world" => 2}, "<unk>")

    assert Llamex.Tokenizer.encode(tokenizer, "hello missing world") == [1, 0, 2]
    assert Llamex.Tokenizer.decode(tokenizer, [1, 2]) == "hello world"
  end

  test "encodes text with the minimal bpe tokenizer" do
    tokenizer =
      Llamex.Tokenizer.bpe(
        %{"<unk>" => 0, "l" => 1, "o" => 2, "w" => 3, "lo" => 4, "low" => 5},
        [["l", "o"], ["lo", "w"]],
        "<unk>"
      )

    assert Llamex.Tokenizer.encode(tokenizer, "low") == [5]
    assert Llamex.Tokenizer.decode(tokenizer, [5]) == "low"
  end

  test "runs one transformer-style attention layer and stores kv cache" do
    model =
      Llamex.new_model(%{
        config: %{vocab_size: 2, embedding_size: 2},
        token_embeddings: %{
          0 => [1.0, 0.0],
          1 => [0.0, 1.0]
        },
        layers: [
          %{
            attention_norm: [1.0, 1.0],
            wq: [[1.0, 0.0], [0.0, 1.0]],
            wk: [[1.0, 0.0], [0.0, 1.0]],
            wv: [[1.0, 0.0], [0.0, 1.0]],
            wo: [[1.0, 0.0], [0.0, 1.0]]
          }
        ],
        output: %{weight: [[1.0, 0.0], [0.0, 1.0]]}
      })

    context = Llamex.new_context(model, Llamex.Backend.List)

    {context, next_token} = Llamex.next_token(context, 0)

    assert next_token == 0
    assert context.tokens == [0]
    assert [{_key, _value}] = Llamex.KVCache.entries(context.kv_cache, 0)
  end

  test "applies RoPE without changing vectors at position zero" do
    assert Llamex.Layers.RoPE.apply([1.0, 2.0], 0, 10_000.0) == [1.0, 2.0]
  end

  test "runs a transformer block with SwiGLU feed-forward weights" do
    model =
      Llamex.new_model(%{
        config: %{vocab_size: 2, embedding_size: 2},
        token_embeddings: %{
          0 => [1.0, 0.0],
          1 => [0.0, 1.0]
        },
        layers: [
          %{
            attention_norm: [1.0, 1.0],
            feed_forward_norm: [1.0, 1.0],
            wq: [[1.0, 0.0], [0.0, 1.0]],
            wk: [[1.0, 0.0], [0.0, 1.0]],
            wv: [[1.0, 0.0], [0.0, 1.0]],
            wo: [[1.0, 0.0], [0.0, 1.0]],
            w_gate: [[1.0, 0.0], [0.0, 1.0]],
            w_up: [[1.0, 0.0], [0.0, 1.0]],
            w_down: [[1.0, 0.0], [0.0, 1.0]]
          }
        ],
        output: %{weight: [[1.0, 0.0], [0.0, 1.0]]}
      })

    context = Llamex.new_context(model, Llamex.Backend.List)

    {context, next_token} = Llamex.next_token(context, 0)

    assert next_token == 0
    assert context.tokens == [0]
    assert [{_key, _value}] = Llamex.KVCache.entries(context.kv_cache, 0)
  end

  test "runs multi-head attention" do
    model =
      Llamex.new_model(%{
        config: %{vocab_size: 2, embedding_size: 4},
        token_embeddings: %{
          0 => [1.0, 0.0, 0.0, 1.0],
          1 => [0.0, 1.0, 1.0, 0.0]
        },
        layers: [
          %{
            head_count: 2,
            attention_norm: [1.0, 1.0, 1.0, 1.0],
            wq: identity4(),
            wk: identity4(),
            wv: identity4(),
            wo: identity4()
          }
        ],
        output: %{weight: identity4() |> Enum.take(2)}
      })

    context = Llamex.new_context(model, Llamex.Backend.List)

    {context, _next_token} = Llamex.next_token(context, 0)

    assert context.tokens == [0]
    assert [{keys, values}] = Llamex.KVCache.entries(context.kv_cache, 0)
    assert length(keys) == 2
    assert length(values) == 2
  end

  test "samples with temperature and top-k" do
    logits = Llamex.Backend.List.from_list([0.0, 1.0, 2.0])

    assert Llamex.Sampler.sample(logits, Llamex.Backend.List, %{
             temperature: 1.0,
             top_k: 1,
             random: 0.0
           }) == 2
  end

  test "samples with top-p" do
    logits = Llamex.Backend.List.from_list([0.0, 1.0, 2.0])

    assert Llamex.Sampler.sample(logits, Llamex.Backend.List, %{
             temperature: 1.0,
             top_p: 0.5,
             random: 0.0
           }) == 2
  end

  test "applies repetition penalty before sampling" do
    logits = Llamex.Backend.List.from_list([3.0, 2.9, 0.0])

    assert Llamex.Sampler.sample(logits, Llamex.Backend.List, %{
             temperature: 1.0,
             top_k: 1,
             repetition_penalty: 2.0,
             history: [0],
             random: 0.0
           }) == 1
  end

  test "generates with seed-based sampling" do
    tokenizer = Llamex.Tokenizer.new(%{"<unk>" => 0, "hello" => 1, "world" => 2}, "<unk>")

    model =
      Llamex.new_model(%{
        config: %{vocab_size: 3, embedding_size: 2},
        tokenizer: tokenizer,
        token_embeddings: %{
          0 => [0.0, 0.0],
          1 => [1.0, 0.0],
          2 => [2.0, 0.0]
        }
      })

    result =
      Llamex.generate(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 2,
        stop_token: 2,
        sampler: %{
          temperature: 1.0,
          top_k: 1,
          seed: 42
        }
      })

    assert result.generated_tokens == [2]
    assert result.text == "world"
  end

  test "generates ordinary text from a prompt" do
    tokenizer = Llamex.Tokenizer.new(%{"<unk>" => 0, "hello" => 1, "world" => 2}, "<unk>")

    model =
      Llamex.new_model(%{
        config: %{vocab_size: 3, embedding_size: 2},
        tokenizer: tokenizer,
        token_embeddings: %{
          0 => [0.0, 0.0],
          1 => [1.0, 0.0],
          2 => [2.0, 0.0]
        }
      })

    result =
      Llamex.generate(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 2,
        stop_token: 2
      })

    assert result.prompt_tokens == [1]
    assert result.generated_tokens == [2]
    assert result.text == "world"
    assert result.context.tokens == [1]
  end

  test "loads a tiny model from json" do
    model = Llamex.ModelLoader.load_json("priv/models/tiny.json")

    result =
      Llamex.generate(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 2,
        stop_token: model.tokenizer.token_to_id["world"]
      })

    assert result.text == "world"
    assert result.prompt_tokens == [1]
    assert result.generated_tokens == [2]
  end

  test "loads a tiny bpe tokenizer from json" do
    model = Llamex.ModelLoader.load_json("priv/models/tiny_bpe.json")

    assert Llamex.encode(model, "low") == [5]
    assert Llamex.decode(model, [5]) == "low"
  end

  test "loads a tokenizer.json bpe tokenizer" do
    tokenizer = Llamex.Tokenizer.Loader.load_tokenizer_json("priv/tokenizers/tiny_tokenizer.json")

    assert Llamex.Tokenizer.encode(tokenizer, "low") == [5]
    assert Llamex.Tokenizer.decode(tokenizer, [5]) == "low"
  end

  test "loads a model with tokenizer.json path" do
    model = Llamex.ModelLoader.load_json("priv/models/tiny_tokenizer_file.json")

    assert Llamex.encode(model, "low") == [5]
  end

  test "loads token embeddings from named tensors" do
    model = Llamex.ModelLoader.load_json("priv/models/tiny_tensors.json")

    result =
      Llamex.generate(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 2,
        stop_token: model.tokenizer.token_to_id["world"]
      })

    assert result.text == "world"
    assert result.generated_tokens == [2]
  end

  test "loads transformer layer and output weights from named tensors" do
    model = Llamex.ModelLoader.load_json("priv/models/tiny_transformer_tensors.json")

    assert [%{wq: wq, attention_norm: attention_norm}] = model.layers
    assert wq == [[1.0, 0.0], [0.0, 1.0]]
    assert attention_norm == [1.0, 1.0]
    assert model.output == %{weight: [[1.0, 0.0], [0.0, 1.0]]}

    context = Llamex.new_context(model, Llamex.Backend.List)
    {context, next_token} = Llamex.next_token(context, 0)

    assert context.tokens == [0]
    assert next_token == 0
  end

  defp identity4 do
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]
  end
end
