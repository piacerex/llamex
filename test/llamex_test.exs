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

  test "reads gguf metadata and tensor directory" do
    gguf = tiny_gguf(:without_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert parsed.version == 3
    assert parsed.tensor_count == 1
    assert parsed.metadata_count == 9
    assert parsed.metadata["general.architecture"] == %{type: :string, value: "llama"}
    assert parsed.metadata["general.alignment"] == %{type: :uint32, value: 32}

    assert parsed.metadata["tokenizer.ggml.tokens"] == %{
             type: :array,
             value: %{type: :string, values: ["<unk>", "hello"]}
           }

    assert [
             %{
               name: "token_embd.weight",
               dimensions: [2, 2],
               type: 0,
               offset: 0
             }
           ] = parsed.tensors

    assert rem(parsed.tensor_data_offset, 32) == 0
  end

  test "builds a tokenizer from gguf metadata" do
    parsed = Llamex.GGUF.Reader.read_binary(tiny_gguf(:without_tensor_data))

    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(parsed.metadata)

    assert Llamex.Tokenizer.encode(tokenizer, "hello missing") == [1, 0]
    assert Llamex.Tokenizer.decode(tokenizer, [0, 1]) == "<unk> hello"
  end

  test "builds a bpe tokenizer from gguf metadata merges" do
    parsed = Llamex.GGUF.Reader.read_binary(tiny_bpe_gguf())

    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(parsed.metadata)

    assert Llamex.Tokenizer.encode(tokenizer, "low") == [5]
    assert Llamex.Tokenizer.decode(tokenizer, [5]) == "low"
  end

  test "reads gguf metadata from a file path" do
    path = Path.join(System.tmp_dir!(), "llamex-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_gguf(:without_tensor_data))

      parsed = Llamex.GGUF.Reader.read_metadata(path)

      assert parsed.metadata["general.architecture"].value == "llama"
    after
      File.rm(path)
    end
  end

  test "reads f32 gguf tensor data into named tensor schema" do
    gguf = tiny_gguf(:with_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors == %{
             "token_embd.weight" => %{
               "shape" => [2, 2],
               "dtype" => "f32",
               "data" => [1.0, 0.0, 0.0, 1.0]
             }
           }
  end

  test "normalizes rank-2 gguf dimensions into llamex schema shape" do
    gguf = tiny_gguf(:with_rectangular_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [2, 3]
    assert tensors["token_embd.weight"]["data"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  end

  test "reads f16 gguf tensor data into named tensor schema" do
    gguf = tiny_gguf(:with_f16_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors == %{
             "token_embd.weight" => %{
               "shape" => [2, 2],
               "dtype" => "f16",
               "data" => [1.0, 0.0, 0.0, -2.0]
             }
           }
  end

  test "reads f32 gguf tensor data from a file path" do
    path =
      Path.join(System.tmp_dir!(), "llamex-tensor-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_gguf(:with_tensor_data))

      tensors = Llamex.GGUF.Reader.read_tensors(path)

      assert tensors["token_embd.weight"]["data"] == [1.0, 0.0, 0.0, 1.0]
    after
      File.rm(path)
    end
  end

  test "loads a Llamex model from a f32 gguf file" do
    path =
      Path.join(System.tmp_dir!(), "llamex-model-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_gguf(:with_tensor_data))

      model = Llamex.GGUF.ModelLoader.load(path)

      assert Llamex.encode(model, "hello") == [1]
      assert model.config.vocab_size == 2
      assert model.config.embedding_size == 2
      assert model.config.context_size == 16
      assert model.config.block_count == 1
      assert model.config.attention_head_count == 2
      assert model.config.attention_head_count_kv == 1
      assert model.config.feed_forward_size == 8
      assert model.token_embeddings == %{0 => [1.0, 0.0], 1 => [0.0, 1.0]}

      result =
        Llamex.generate(model, "<unk>", %{
          backend: Llamex.Backend.List,
          max_new_tokens: 1,
          stop_token: nil
        })

      assert result.generated_tokens == [0]
    after
      File.rm(path)
    end
  end

  defp identity4 do
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]
  end

  defp tiny_gguf(mode) do
    header = [
      "GGUF",
      u32(3),
      u64(1),
      u64(9)
    ]

    metadata = [
      kv_string("general.architecture", "llama"),
      kv_u32("general.alignment", 32),
      kv_u32("llama.embedding_length", 2),
      kv_u32("llama.context_length", 16),
      kv_u32("llama.block_count", 1),
      kv_u32("llama.attention.head_count", 2),
      kv_u32("llama.attention.head_count_kv", 1),
      kv_u32("llama.feed_forward_length", 8),
      kv_array_string("tokenizer.ggml.tokens", ["<unk>", "hello"])
    ]

    {dimensions, tensor_type, values} =
      case mode do
        :with_rectangular_tensor_data -> {[3, 2], 0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        :with_f16_tensor_data -> {[2, 2], 1, [0x3C00, 0x0000, 0x0000, 0xC000]}
        _other -> {[2, 2], 0, [1.0, 0.0, 0.0, 1.0]}
      end

    tensor_infos = [
      gguf_string("token_embd.weight"),
      u32(length(dimensions)),
      Enum.map(dimensions, &u64/1),
      u32(tensor_type),
      u64(0)
    ]

    without_data = IO.iodata_to_binary([header, metadata, tensor_infos])

    case mode do
      :without_tensor_data -> without_data
      :with_tensor_data -> with_aligned_f32_tensor_data(without_data, values)
      :with_rectangular_tensor_data -> with_aligned_f32_tensor_data(without_data, values)
      :with_f16_tensor_data -> with_aligned_f16_tensor_data(without_data, values)
    end
  end

  defp with_aligned_f32_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    tensor_data = Enum.map(values, fn value -> <<value::little-float-size(32)>> end)

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp with_aligned_f16_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    tensor_data = Enum.map(values, fn value -> <<value::little-unsigned-integer-size(16)>> end)

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp tiny_bpe_gguf do
    header = [
      "GGUF",
      u32(3),
      u64(0),
      u64(4)
    ]

    metadata = [
      kv_string("general.architecture", "llama"),
      kv_array_string("tokenizer.ggml.tokens", ["<unk>", "l", "o", "w", "lo", "low"]),
      kv_array_string("tokenizer.ggml.merges", ["l o", "lo w"]),
      kv_u32("tokenizer.ggml.unknown_token_id", 0)
    ]

    IO.iodata_to_binary([header, metadata])
  end

  defp kv_string(key, value), do: [gguf_string(key), u32(8), gguf_string(value)]
  defp kv_u32(key, value), do: [gguf_string(key), u32(4), u32(value)]

  defp kv_array_string(key, values) do
    [gguf_string(key), u32(9), u32(8), u64(length(values)), Enum.map(values, &gguf_string/1)]
  end

  defp gguf_string(value) do
    [u64(byte_size(value)), value]
  end

  defp u32(value), do: <<value::little-unsigned-integer-size(32)>>
  defp u64(value), do: <<value::little-unsigned-integer-size(64)>>
end
