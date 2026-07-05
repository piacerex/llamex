defmodule LlamexTest do
  use ExUnit.Case
  import ExUnit.CaptureIO

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

  test "encodes sentencepiece word starts when available" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{"<unk>" => 0, "Hello" => 1, "▁Hello" => 2, "world" => 3, "▁world" => 4},
        "<unk>"
      )

    assert Llamex.Tokenizer.encode(tokenizer, "Hello world") == [2, 4]
  end

  test "encodes sentencepiece words with longest subword pieces" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{"<unk>" => 0, "▁bind" => 1, "ing" => 2, "▁pon" => 3},
        "<unk>"
      )

    assert tokenizer.sentencepiece_vocab?
    assert Enum.map(tokenizer.tokens_by_length, &byte_size/1) == [7, 6, 5, 3]
    assert Llamex.Tokenizer.encode(tokenizer, "binding pon") == [1, 2, 3]
  end

  test "encodes chat template special tokens before byte fallback" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{
          "<unk>" => 0,
          "<|im_start|>" => 1,
          "<|im_end|>" => 2,
          "user" => 3,
          "Hi" => 4
        },
        "<unk>"
      )

    assert Llamex.Tokenizer.encode(tokenizer, "<|im_start|>user Hi<|im_end|>") == [1, 3, 4, 2]
  end

  test "detects chat template markers missing from tokenizer vocab" do
    assert Llamex.ChatTemplate.markers(chatml_template()) == ["<|im_start|>", "<|im_end|>"]

    assert Llamex.ChatTemplate.missing_tokens(chatml_template(), %{
             "<unk>" => 0,
             "<|im_start|>" => 1
           }) == ["<|im_end|>"]
  end

  test "applies role marker chat templates with tokenizer eos token" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{"<unk>" => 0, "</s>" => 1, "<|user|>" => 2, "<|assistant|>" => 3},
        "<unk>",
        special_tokens: %{eos: %{id: 1, token: "</s>"}}
      )

    assert Llamex.ChatTemplate.supported?(role_marker_template())

    assert Llamex.ChatTemplate.apply(role_marker_template(), "Hello", tokenizer) ==
             "<|user|>\nHello</s><|assistant|>"
  end

  test "adds configured bos and eos tokens while encoding" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{"<unk>" => 0, "<s>" => 1, "</s>" => 2, "hello" => 3},
        "<unk>",
        special_tokens: %{
          bos: %{id: 1, token: "<s>"},
          eos: %{id: 2, token: "</s>"},
          add_bos: true,
          add_eos: true
        }
      )

    assert Llamex.Tokenizer.encode(tokenizer, "hello") == [1, 3, 2]
    assert Llamex.Tokenizer.encode(tokenizer, "<s> hello </s>") == [1, 3, 2]
  end

  test "decodes sentencepiece-style gguf tokens as plain text" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{"<unk>" => 0, "<s>" => 1, "▁Hello" => 2, "▁world" => 3, "!" => 4},
        "<unk>",
        token_types: [
          %{id: 0, token: "<unk>", type: :unknown, type_id: 2},
          %{id: 1, token: "<s>", type: :control, type_id: 3},
          %{id: 2, token: "▁Hello", type: :normal, type_id: 1},
          %{id: 3, token: "▁world", type: :normal, type_id: 1},
          %{id: 4, token: "!", type: :normal, type_id: 1}
        ]
      )

    assert Llamex.Tokenizer.decode(tokenizer, [1, 2, 3, 4]) == "Hello world!"
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

  test "backend list dot checks vector lengths in one pass" do
    assert Llamex.Backend.List.dot([1.0, 2.0], [3.0, 4.0]) == 11.0

    assert_raise ArgumentError, ~r/vectors must have matching lengths/, fn ->
      Llamex.Backend.List.dot([1.0], [1.0, 2.0])
    end
  end

  test "runs large matvec with the same row order" do
    vector = List.duplicate(1.0, 1000)
    rows = Enum.map(1..1001, fn value -> [value / 1000.0 | List.duplicate(0.0, 999)] end)

    assert Llamex.Tensor.matvec(rows, vector) == Enum.map(1..1001, &(&1 / 1000.0))
  end

  test "finds argmax matvec without materializing logits" do
    vector = [1.0, 2.0, 3.0]

    assert Llamex.Tensor.argmax_matvec(
             [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 2.0, 0.0]],
             vector
           ) == 1
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

  test "profiles SwiGLU feed-forward substeps" do
    model =
      Llamex.new_model(%{
        config: %{vocab_size: 2, embedding_size: 2},
        tokenizer: Llamex.Tokenizer.new(%{"<unk>" => 0, "hello" => 1}, "<unk>"),
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

    profile = Llamex.Profile.generation_step(model, "hello", %{backend: Llamex.Backend.List})

    [layer] = profile.eval_timings.layers
    mlp = Enum.find(layer.components, &(&1.label == "mlp"))

    assert Enum.map(mlp.components, & &1.label) == [
             "feed_forward_norm",
             "w_gate",
             "w_up",
             "silu_multiply",
             "w_down",
             "residual"
           ]
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

  test "runs grouped-query attention with fewer kv heads" do
    model =
      Llamex.new_model(%{
        config: %{vocab_size: 2, embedding_size: 4},
        token_embeddings: %{
          0 => [1.0, 0.0, 0.0, 1.0],
          1 => [0.0, 1.0, 1.0, 0.0]
        },
        layers: [
          %{
            head_count: 4,
            kv_head_count: 2,
            attention_norm: [1.0, 1.0, 1.0, 1.0],
            wq: identity4(),
            wk: [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            wv: [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            wo: identity4()
          }
        ],
        output: %{weight: identity4() |> Enum.take(2)}
      })

    context = Llamex.new_context(model, Llamex.Backend.List)

    {context, _next_token} = Llamex.next_token(context, 0)

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

  test "reports sampled candidate probabilities" do
    logits = Llamex.Backend.List.from_list([0.0, 1.0, 2.0])

    candidates =
      Llamex.Sampler.candidates(
        logits,
        Llamex.Backend.List,
        %{
          temperature: 1.0,
          top_k: 2
        },
        2
      )

    assert Enum.map(candidates, & &1.token) == [2, 1]
    assert Enum.all?(candidates, &is_float(&1.probability))
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
    assert result.finish_reason == :stop
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
    assert result.finish_reason == :stop
    assert result.text == "world"
    assert result.context.tokens == [1]
  end

  test "generates until any configured stop token" do
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
        stop_tokens: [0, 2]
      })

    assert result.generated_tokens == [2]
    assert result.finish_reason == :stop
  end

  test "prefills and steps generation with a loaded model" do
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

    state = Llamex.prefill(model, "hello", %{backend: Llamex.Backend.List})
    step = Llamex.step(state.context, state.current_token, %{sampler: :greedy})

    assert state.prompt_tokens == [1]
    assert step.token == 2
    assert step.text == "world"
    assert step.context.tokens == [1]
  end

  test "nx backend can generate through prepared output weights" do
    if Code.ensure_loaded?(Nx) do
      tokenizer = Llamex.Tokenizer.new(%{"<unk>" => 0, "hello" => 1, "world" => 2}, "<unk>")

      model =
        Llamex.new_model(%{
          config: %{vocab_size: 3, embedding_size: 2},
          tokenizer: tokenizer,
          token_embeddings: %{
            0 => [0.0, 0.0],
            1 => [1.0, 0.0],
            2 => [2.0, 0.0]
          },
          output: %{weight: [[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]]}
        })

      result =
        Llamex.generate(model, "hello", %{
          backend: Llamex.Backend.Nx,
          max_new_tokens: 1,
          stop_tokens: [2]
        })

      assert result.generated_tokens == [2]
      assert result.finish_reason == :stop
    end
  end

  test "profiles one generation step" do
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

    profile = Llamex.Profile.generation_step(model, "hello", %{backend: Llamex.Backend.List})

    assert profile.prompt_tokens == 1
    assert profile.token == 2
    assert profile.text == "world"
    assert profile.eval_timings.layers == []
    assert profile.eval_timings.output_norm.label == "output_norm"
    assert profile.eval_timings.logits.label == "logits"

    assert Enum.map(profile.prefill_timings, & &1.label) == [
             "prompt_encode",
             "backend_prepare",
             "prompt_eval"
           ]

    assert Enum.map(profile.timings, & &1.label) == ["prefill", "step"]
    assert Enum.all?(profile.timings, &is_integer(&1.milliseconds))
  end

  test "profiles prefill tokens individually" do
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

    profile = Llamex.Profile.prefill_steps(model, "hello world", %{backend: Llamex.Backend.List})

    assert profile.prompt_tokens == [1, 2]
    assert profile.current_token == 2
    assert profile.current_piece == "world"
    assert profile.context_tokens == [1]
    assert Enum.map(profile.steps, & &1.token) == [1]
    assert Enum.map(profile.steps, & &1.piece) == ["hello"]
    assert Enum.map(profile.steps, & &1.timing.label) == ["prefill_1"]
  end

  test "profiles multiple generation steps" do
    tokenizer =
      Llamex.Tokenizer.whitespace(
        %{"<unk>" => 0, "hello" => 1, "world" => 2},
        "<unk>",
        token_types: [
          %{id: 0, token: "<unk>", type: :unknown, type_id: 2},
          %{id: 1, token: "hello", type: :normal, type_id: 1},
          %{id: 2, token: "world", type: :normal, type_id: 1}
        ]
      )

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

    profile =
      Llamex.Profile.generation_steps(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 2
      })

    assert profile.prompt_tokens == 1
    assert profile.prompt_token_ids == [1]
    assert profile.prompt_pieces == ["hello"]
    assert profile.backend == Llamex.Backend.List
    assert profile.max_new_tokens == 2
    assert profile.stop_token == nil
    assert profile.stop_tokens == []

    assert Enum.map(profile.prefill_timings, & &1.label) == [
             "prompt_encode",
             "backend_prepare",
             "prompt_eval"
           ]

    assert profile.sampler == :greedy
    assert profile.generated_tokens == [2, 2]
    assert profile.generated_pieces == ["world", "world"]

    assert profile.generated_token_info == [
             %{token: 2, piece: "world", type: :normal, type_id: 1},
             %{token: 2, piece: "world", type: :normal, type_id: 1}
           ]

    assert profile.finish_reason == :length
    assert profile.text == "world world"
    assert Enum.map(profile.steps, & &1.token) == [2, 2]
    assert Enum.map(profile.steps, & &1.piece) == ["world", "world"]
    assert Enum.map(profile.steps, & &1.eval_timings.logits.label) == ["logits", "logits"]

    assert Enum.map(profile.steps, &Map.take(&1, [:token, :piece, :type, :type_id])) == [
             %{token: 2, piece: "world", type: :normal, type_id: 1},
             %{token: 2, piece: "world", type: :normal, type_id: 1}
           ]

    assert Enum.map(profile.timings, & &1.label) == ["prefill", "step_1", "step_2"]
  end

  test "profiles generation steps until stop token" do
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

    profile =
      Llamex.Profile.generation_steps(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 4,
        stop_token: 2
      })

    assert profile.generated_tokens == [2]
    assert profile.finish_reason == :stop
    assert profile.text == "world"
    assert Enum.map(profile.timings, & &1.label) == ["prefill", "step_1"]
  end

  test "profiles multiple sampled generation steps" do
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

    profile =
      Llamex.Profile.generation_steps(model, "hello", %{
        backend: Llamex.Backend.List,
        max_new_tokens: 2,
        sampler: %{temperature: 1.0, top_k: 1, seed: 1}
      })

    assert Enum.map(profile.steps, & &1.token) == [2, 2]
    assert profile.sampler == %{temperature: 1.0, top_k: 1, seed: 1}
  end

  test "generate task rejects chat templates with missing tokenizer tokens" do
    path = Path.join(System.tmp_dir!(), "llamex-chat-#{System.unique_integer([:positive])}.json")

    model = %{
      "config" => %{"vocab_size" => 2, "embedding_size" => 2},
      "tokenizer" => %{
        "type" => "whitespace",
        "unknown_token" => "<unk>",
        "chat_template" => chatml_template(),
        "vocab" => %{"<unk>" => 0, "hello" => 1}
      },
      "token_embeddings" => %{
        "0" => [0.0, 0.0],
        "1" => [1.0, 0.0]
      }
    }

    try do
      File.write!(path, JSON.encode!(model))

      assert_raise Mix.Error, ~r/chat template references missing tokenizer tokens/, fn ->
        Mix.Tasks.Llamex.Generate.run([path, "hello", "1", "--chat"])
      end
    after
      File.rm(path)
    end
  end

  test "generate task validates gguf chat templates before loading tensors" do
    path = Path.join(System.tmp_dir!(), "llamex-chat-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_chat_template_gguf())

      assert_raise Mix.Error, ~r/chat template references missing tokenizer tokens/, fn ->
        Mix.Tasks.Llamex.Generate.run([path, "hello", "1", "--chat"])
      end
    after
      File.rm(path)
    end
  end

  test "generate task can print a generation profile" do
    output =
      capture_io(fn ->
        Mix.Tasks.Llamex.Generate.run(["priv/models/tiny.json", "hello", "2", "--profile"])
      end)

    profile = JSON.decode!(String.trim(output))

    assert profile["model_path"] == "priv/models/tiny.json"
    assert profile["original_prompt"] == "hello"
    assert profile["prompt"] == "hello"
    assert profile["prompt_tokens"] == 1
    assert profile["prompt_token_ids"] == [1]
    assert profile["prompt_pieces"] == ["hello"]
    assert profile["backend"] == "Elixir.Llamex.Backend.List"
    assert profile["max_new_tokens"] == 2
    assert profile["sampler"] == "greedy"
    assert profile["generated_tokens"] == [2]
    assert profile["generated_pieces"] == ["world"]
    assert profile["generated_token_info"] == [%{"token" => 2, "piece" => "world"}]
    assert profile["finish_reason"] == "stop"
    assert profile["text"] == "world"

    assert Enum.map(profile["steps"], &get_in(&1, ["eval_timings", "logits", "label"])) == [
             "logits"
           ]

    assert Enum.map(profile["prefill_timings"], & &1["label"]) == [
             "prompt_encode",
             "backend_prepare",
             "prompt_eval"
           ]

    assert Enum.map(profile["steps"], & &1["piece"]) == ["world"]
    assert Enum.map(profile["timings"], & &1["label"]) == ["prefill", "step_1"]
  end

  test "generate task can print candidate tokens in a generation profile" do
    output =
      capture_io(fn ->
        Mix.Tasks.Llamex.Generate.run([
          "priv/models/tiny.json",
          "hello",
          "1",
          "--profile",
          "--candidates",
          "2"
        ])
      end)

    profile = JSON.decode!(String.trim(output))

    assert [step] = profile["steps"]
    assert Enum.map(step["candidates"], & &1["piece"]) == ["world", "hello"]
  end

  test "generate task can disable inferred stop token for profiling" do
    output =
      capture_io(fn ->
        Mix.Tasks.Llamex.Generate.run([
          "priv/models/tiny.json",
          "hello",
          "2",
          "--profile",
          "--no-stop"
        ])
      end)

    profile = JSON.decode!(String.trim(output))

    assert profile["generated_tokens"] == [2, 2]
    assert profile["finish_reason"] == "length"
  end

  test "generate task can use an explicit stop token" do
    output =
      capture_io(fn ->
        Mix.Tasks.Llamex.Generate.run([
          "priv/models/tiny.json",
          "hello",
          "2",
          "--profile",
          "--stop-token",
          "2"
        ])
      end)

    profile = JSON.decode!(String.trim(output))

    assert profile["generated_tokens"] == [2]
    assert profile["finish_reason"] == "stop"
  end

  test "generate task can use an explicit stop piece" do
    output =
      capture_io(fn ->
        Mix.Tasks.Llamex.Generate.run([
          "priv/models/tiny.json",
          "hello",
          "2",
          "--profile",
          "--stop-piece",
          "world"
        ])
      end)

    profile = JSON.decode!(String.trim(output))

    assert profile["generated_tokens"] == [2]
    assert profile["finish_reason"] == "stop"
  end

  test "generate task rejects unknown stop pieces" do
    assert_raise Mix.Error, ~r/stop piece not found in tokenizer vocab: missing/, fn ->
      Mix.Tasks.Llamex.Generate.run([
        "priv/models/tiny.json",
        "hello",
        "2",
        "--profile",
        "--stop-piece",
        "missing"
      ])
    end
  end

  test "generate task can use an explicit special stop token" do
    path =
      Path.join(
        System.tmp_dir!(),
        "llamex-stop-special-#{System.unique_integer([:positive])}.json"
      )

    model = %{
      "config" => %{"vocab_size" => 3, "embedding_size" => 2},
      "tokenizer" => %{
        "type" => "whitespace",
        "unknown_token" => "<unk>",
        "vocab" => %{"<unk>" => 0, "hello" => 1, "world" => 2},
        "special_tokens" => %{"eos" => %{"id" => 2, "token" => "world"}}
      },
      "token_embeddings" => %{
        "0" => [0.0, 0.0],
        "1" => [1.0, 0.0],
        "2" => [2.0, 0.0]
      }
    }

    try do
      File.write!(path, JSON.encode!(model))

      output =
        capture_io(fn ->
          Mix.Tasks.Llamex.Generate.run([
            path,
            "hello",
            "2",
            "--profile",
            "--stop-special",
            "eos"
          ])
        end)

      profile = JSON.decode!(String.trim(output))

      assert profile["generated_tokens"] == [2]
      assert profile["stop_tokens"] == [2]
      assert profile["finish_reason"] == "stop"
    after
      File.rm(path)
    end
  end

  test "generate task can stop on generated control tokens" do
    path =
      Path.join(
        System.tmp_dir!(),
        "llamex-stop-control-#{System.unique_integer([:positive])}.json"
      )

    model = %{
      "config" => %{"vocab_size" => 3, "embedding_size" => 2},
      "tokenizer" => %{
        "type" => "whitespace",
        "unknown_token" => "<unk>",
        "vocab" => %{"<unk>" => 0, "hello" => 1, "<ctrl>" => 2},
        "token_types" => [
          %{"id" => 0, "token" => "<unk>", "type" => "unknown", "type_id" => 2},
          %{"id" => 1, "token" => "hello", "type" => "normal", "type_id" => 1},
          %{"id" => 2, "token" => "<ctrl>", "type" => "control", "type_id" => 3}
        ]
      },
      "token_embeddings" => %{
        "0" => [0.0, 0.0],
        "1" => [1.0, 0.0],
        "2" => [2.0, 0.0]
      }
    }

    try do
      File.write!(path, JSON.encode!(model))

      output =
        capture_io(fn ->
          Mix.Tasks.Llamex.Generate.run([
            path,
            "hello",
            "3",
            "--profile",
            "--stop-control"
          ])
        end)

      profile = JSON.decode!(String.trim(output))

      assert profile["generated_tokens"] == [2]

      assert profile["generated_token_info"] == [
               %{"token" => 2, "piece" => "<ctrl>", "type" => "control", "type_id" => 3}
             ]

      assert profile["stop_tokens"] == [2]
      assert profile["finish_reason"] == "stop"
    after
      File.rm(path)
    end
  end

  test "generate task rejects unknown special stop token names" do
    assert_raise Mix.Error, ~r/unsupported special stop token: nope/, fn ->
      Mix.Tasks.Llamex.Generate.run([
        "priv/models/tiny.json",
        "hello",
        "2",
        "--profile",
        "--stop-special",
        "nope"
      ])
    end
  end

  test "tokenize task prints token ids and pieces" do
    output =
      capture_io(fn ->
        Mix.Tasks.Llamex.Tokenize.run(["priv/models/tiny.json", "hello world"])
      end)

    result = JSON.decode!(String.trim(output))

    assert result["prompt"] == "hello world"
    assert result["token_count"] == 2

    assert result["tokens"] == [
             %{"id" => 1, "piece" => "hello"},
             %{"id" => 2, "piece" => "world"}
           ]
  end

  test "tokenize task prints gguf token type metadata" do
    path =
      Path.join(System.tmp_dir!(), "llamex-tokenize-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_byte_token_gguf())

      output =
        capture_io(fn ->
          Mix.Tasks.Llamex.Tokenize.run([path, "hi"])
        end)

      result = JSON.decode!(String.trim(output))

      assert result["tokens"] == [
               %{"id" => 1, "piece" => "<0x68>", "type" => "byte", "type_id" => 6},
               %{"id" => 2, "piece" => "<0x69>", "type" => "byte", "type_id" => 6}
             ]
    after
      File.rm(path)
    end
  end

  test "tokenize task validates gguf chat templates before tensor loading" do
    path =
      Path.join(System.tmp_dir!(), "llamex-tokenize-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_chat_template_gguf())

      assert_raise Mix.Error, ~r/chat template references missing tokenizer tokens/, fn ->
        Mix.Tasks.Llamex.Tokenize.run([path, "hello", "--chat"])
      end
    after
      File.rm(path)
    end
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
    assert model.output_norm == [1.0, 1.0]
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

  test "diagnoses unsupported gguf tensor types without reading tensor data" do
    diagnostic = Llamex.GGUF.Diagnostic.inspect_binary(tiny_gguf(:with_unsupported_tensor_type))

    assert diagnostic.version == 3
    assert diagnostic.tensor_count == 1
    assert diagnostic.tensor_element_count == 4
    assert diagnostic.eager_f32_bytes == 16
    assert diagnostic.supported_tensor_types == %{}
    assert diagnostic.unsupported_tensor_types == %{"type_99" => 1}
    assert diagnostic.chat_template == "none"
    assert diagnostic.chat_usable == false
    assert diagnostic.special_tokens == %{}
    assert diagnostic.missing_chat_template_tokens == []

    assert diagnostic.unsupported_tensors == [
             %{name: "token_embd.weight", type: 99, dimensions: [2, 2]}
           ]

    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~
             "unsupported tensors:\n- token_embd.weight: type_99 [2, 2]"

    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~ "chat template missing tokens: none"
    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~ "tensor elements: 4"
    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~ "eager f32 lower bound: 16 B"
  end

  test "gguf inspect task can print json diagnostics" do
    path =
      Path.join(System.tmp_dir!(), "llamex-inspect-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(path, tiny_gguf(:with_unsupported_tensor_type))

      output =
        capture_io(fn ->
          Mix.Tasks.Llamex.Gguf.Inspect.run([path, "--json"])
        end)

      [diagnostic] = JSON.decode!(String.trim(output))

      assert diagnostic["path"] == path
      assert diagnostic["version"] == 3
      assert diagnostic["chat_template"] == "none"
      assert diagnostic["chat_usable"] == false
      assert diagnostic["special_tokens"] == %{}
      assert diagnostic["unsupported_tensor_types"] == %{"type_99" => 1}
    after
      File.rm(path)
    end
  end

  test "diagnoses gguf special tokens" do
    diagnostic = Llamex.GGUF.Diagnostic.inspect_binary(tiny_special_token_gguf())

    assert diagnostic.special_tokens == %{
             unknown: %{id: 0, piece: "<unk>"},
             bos: %{id: 1, piece: "<s>"},
             eos: %{id: 2, piece: "</s>"}
           }

    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~
             "special tokens: bos=1:<s>, eos=2:</s>, unknown=0:<unk>"
  end

  test "gguf inspect task can print json diagnostics for multiple files" do
    first =
      Path.join(System.tmp_dir!(), "llamex-inspect-a-#{System.unique_integer([:positive])}.gguf")

    second =
      Path.join(System.tmp_dir!(), "llamex-inspect-b-#{System.unique_integer([:positive])}.gguf")

    try do
      File.write!(first, tiny_gguf(:with_unsupported_tensor_type))
      File.write!(second, tiny_chat_template_gguf())

      output =
        capture_io(fn ->
          Mix.Tasks.Llamex.Gguf.Inspect.run([first, second, "--json"])
        end)

      diagnostics = JSON.decode!(String.trim(output))

      assert Enum.map(diagnostics, & &1["path"]) == [first, second]
      assert Enum.map(diagnostics, & &1["chat_template"]) == ["none", "supported"]
      assert Enum.map(diagnostics, & &1["chat_usable"]) == [false, false]
    after
      File.rm(first)
      File.rm(second)
    end
  end

  test "builds a tokenizer from gguf metadata" do
    parsed = Llamex.GGUF.Reader.read_binary(tiny_gguf(:without_tensor_data))

    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(parsed.metadata)

    assert Llamex.Tokenizer.encode(tokenizer, "hello missing") == [1, 0]
    assert Llamex.Tokenizer.decode(tokenizer, [0, 1]) == "<unk> hello"
  end

  test "builds a tokenizer with gguf special token metadata" do
    parsed = Llamex.GGUF.Reader.read_binary(tiny_special_token_gguf())

    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(parsed.metadata)

    assert tokenizer.special_tokens.unknown == %{id: 0, token: "<unk>"}
    assert tokenizer.special_tokens.bos == %{id: 1, token: "<s>"}
    assert tokenizer.special_tokens.eos == %{id: 2, token: "</s>"}
    assert tokenizer.special_tokens.add_bos == true
    assert tokenizer.special_tokens.add_eos == false

    assert tokenizer.token_types == [
             %{id: 0, token: "<unk>", type: :unknown, type_id: 2},
             %{id: 1, token: "<s>", type: :control, type_id: 3},
             %{id: 2, token: "</s>", type: :control, type_id: 3},
             %{id: 3, token: "hello", type: :normal, type_id: 1}
           ]
  end

  test "diagnoses chat templates with missing marker tokens" do
    diagnostic = Llamex.GGUF.Diagnostic.inspect_binary(tiny_chat_template_gguf())

    assert diagnostic.chat_template == "supported"
    assert diagnostic.chat_usable == false
    assert diagnostic.missing_chat_template_tokens == ["<|im_start|>", "<|im_end|>"]

    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~ "chat template: supported"
    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~ "chat usable: false"

    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~
             "chat template missing tokens: <|im_start|>, <|im_end|>"
  end

  test "diagnoses chat templates as usable when markers exist" do
    diagnostic = Llamex.GGUF.Diagnostic.inspect_binary(tiny_usable_chat_template_gguf())

    assert diagnostic.chat_template == "supported"
    assert diagnostic.chat_usable == true
    assert diagnostic.missing_chat_template_tokens == []

    assert Llamex.GGUF.Diagnostic.format(diagnostic) =~ "chat usable: true"
  end

  test "builds a tokenizer with gguf chat template metadata" do
    parsed = Llamex.GGUF.Reader.read_binary(tiny_chat_template_gguf())

    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(parsed.metadata)

    assert tokenizer.chat_template == chatml_template()

    assert Llamex.ChatTemplate.apply(tokenizer.chat_template, "Hello") ==
             "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
  end

  test "decodes gguf byte tokens" do
    parsed = Llamex.GGUF.Reader.read_binary(tiny_byte_token_gguf())

    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(parsed.metadata)

    assert tokenizer.token_types == [
             %{id: 0, token: "<unk>", type: :unknown, type_id: 2},
             %{id: 1, token: "<0x68>", type: :byte, type_id: 6},
             %{id: 2, token: "<0x69>", type: :byte, type_id: 6}
           ]

    assert Llamex.Tokenizer.encode(tokenizer, "hi") == [1, 2]
    assert Llamex.Tokenizer.decode(tokenizer, [1, 2]) == "hi"
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

  test "reads q4_0 gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q4_0_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [32]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [-8.0, -7.0, 0.0, 7.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == 0.0))
  end

  test "rejects q4_0 tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q4_0_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q4_0 tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q4_1 gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q4_1_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [32]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [10.0, 11.0, 18.0, 25.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == 18.0))
  end

  test "rejects q4_1 tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q4_1_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q4_1 tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q5_0 gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q5_0_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [32]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [0.0, 1.0, -8.0, -1.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == -8.0))
  end

  test "rejects q5_0 tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q5_0_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q5_0 tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q5_1 gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q5_1_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [32]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [26.0, 27.0, 18.0, 25.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == 18.0))
  end

  test "rejects q5_1 tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q5_1_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q5_1 tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q8_0 gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q8_0_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [32]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [0.0, 1.0, -2.0, 3.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == 0.0))
  end

  test "rejects q8_0 tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q8_0_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q8_0 tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q8_1 gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q8_1_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [32]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [0.0, 1.0, -2.0, 3.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == 0.0))
  end

  test "rejects q8_1 tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q8_1_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q8_1 tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q2_k gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q2_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [256]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [-2.0, -1.0, 0.0, 1.0]
    assert Enum.drop(tensors["token_embd.weight"]["data"], 4) == List.duplicate(-1.0, 252)
  end

  test "rejects q2_k tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q2_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q2_K tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q3_k gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q3_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [256]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [-4.0, -1.0, 0.0, 3.0]
    assert Enum.drop(tensors["token_embd.weight"]["data"], 4) == List.duplicate(0.0, 252)
  end

  test "rejects q3_k tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q3_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q3_K tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q4_k gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q4_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [256]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [-2.0, -1.0, 6.0, 13.0]
    assert Enum.drop(tensors["token_embd.weight"]["data"], 4) == List.duplicate(6.0, 252)
  end

  test "rejects q4_k tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q4_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q4_K tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q5_k gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q5_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [256]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [-2.0, 15.0, 6.0, 29.0]
    assert Enum.drop(tensors["token_embd.weight"]["data"], 4) == List.duplicate(6.0, 252)
  end

  test "rejects q5_k tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q5_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q5_K tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q6_k gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q6_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [256]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [0.0, 1.0, -16.0, 31.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == -32.0))
  end

  test "rejects q6_k tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q6_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q6_K tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
  end

  test "reads q8_k gguf tensor data into dequantized named tensor schema" do
    gguf = tiny_gguf(:with_q8_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    tensors = Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)

    assert tensors["token_embd.weight"]["shape"] == [256]
    assert tensors["token_embd.weight"]["dtype"] == "f32"
    assert Enum.take(tensors["token_embd.weight"]["data"], 4) == [0.0, 2.0, -4.0, 6.0]
    assert tensors["token_embd.weight"]["data"] |> Enum.drop(4) |> Enum.all?(&(&1 == 0.0))
  end

  test "rejects q8_k tensors whose element count is not block-aligned" do
    gguf = tiny_gguf(:with_unaligned_q8_k_tensor_data)
    parsed = Llamex.GGUF.Reader.read_binary(gguf)

    assert_raise ArgumentError, ~r/Q8_K tensor element count/, fn ->
      Llamex.GGUF.Reader.read_tensor_data(parsed, gguf)
    end
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

  test "loads gguf output norm and output tensors" do
    path =
      Path.join(
        System.tmp_dir!(),
        "llamex-output-model-#{System.unique_integer([:positive])}.gguf"
      )

    try do
      File.write!(path, tiny_gguf_with_output_tensors())

      model = Llamex.GGUF.ModelLoader.load(path)

      assert model.token_embeddings == %{0 => [1.0, 0.0], 1 => [0.0, 1.0]}
      assert model.output_norm == [1.0, 1.0]
      assert model.output == %{weight: [[1.0, 0.0], [0.0, 1.0]]}

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

  test "loads gguf tokenizer special tokens through model loader" do
    path =
      Path.join(
        System.tmp_dir!(),
        "llamex-special-token-model-#{System.unique_integer([:positive])}.gguf"
      )

    try do
      File.write!(path, tiny_gguf_with_special_token_tensors())

      model = Llamex.GGUF.ModelLoader.load(path)

      assert model.tokenizer.special_tokens.bos == %{id: 1, token: "<s>"}
      assert model.tokenizer.special_tokens.eos == %{id: 2, token: "</s>"}
      assert model.tokenizer.special_tokens.add_bos == true

      assert Enum.map(model.tokenizer.token_types, & &1.type) == [
               :unknown,
               :control,
               :control,
               :normal
             ]

      assert Llamex.encode(model, "hello") == [1, 3]
    after
      File.rm(path)
    end
  end

  test "loads a transformer layer from gguf tensors and runs attention" do
    path =
      Path.join(
        System.tmp_dir!(),
        "llamex-transformer-model-#{System.unique_integer([:positive])}.gguf"
      )

    try do
      File.write!(path, tiny_gguf_with_transformer_tensors())

      model = Llamex.GGUF.ModelLoader.load(path)

      assert [layer] = model.layers
      assert layer.attention_norm == [1.0, 1.0]
      assert layer.wq == [[1.0, 0.0], [0.0, 1.0]]
      assert layer.wk == [[1.0, 0.0], [0.0, 1.0]]
      assert layer.wv == [[1.0, 0.0], [0.0, 1.0]]
      assert layer.wo == [[1.0, 0.0], [0.0, 1.0]]

      context = Llamex.new_context(model, Llamex.Backend.List)
      {context, next_token} = Llamex.next_token(context, 0)

      assert context.tokens == [0]
      assert next_token == 0
      assert [{_key, _value}] = Llamex.KVCache.entries(context.kv_cache, 0)
    after
      File.rm(path)
    end
  end

  test "loads feed-forward tensors from gguf and runs swiglu" do
    path =
      Path.join(
        System.tmp_dir!(),
        "llamex-ffn-model-#{System.unique_integer([:positive])}.gguf"
      )

    try do
      File.write!(path, tiny_gguf_with_feed_forward_tensors())

      model = Llamex.GGUF.ModelLoader.load(path)

      assert [layer] = model.layers
      assert layer.feed_forward_norm == [1.0, 1.0]
      assert layer.w_gate == [[1.0, 0.0], [0.0, 1.0]]
      assert layer.w_up == [[1.0, 0.0], [0.0, 1.0]]
      assert layer.w_down == [[1.0, 0.0], [0.0, 1.0]]
      assert model.config.feed_forward_size == 2

      context = Llamex.new_context(model, Llamex.Backend.List)
      {context, next_token} = Llamex.next_token(context, 0)

      assert context.tokens == [0]
      assert next_token == 0
      assert [{_key, _value}] = Llamex.KVCache.entries(context.kv_cache, 0)
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
        :with_q4_0_tensor_data -> {[32], 2, [0, 1, 8, 15 | List.duplicate(8, 28)]}
        :with_unaligned_q4_0_tensor_data -> {[31], 2, [0, 1, 8, 15 | List.duplicate(8, 28)]}
        :with_q4_1_tensor_data -> {[32], 3, [0, 1, 8, 15 | List.duplicate(8, 28)]}
        :with_unaligned_q4_1_tensor_data -> {[31], 3, [0, 1, 8, 15 | List.duplicate(8, 28)]}
        :with_q5_0_tensor_data -> {[32], 6, [16, 17, 8, 15 | List.duplicate(8, 28)]}
        :with_unaligned_q5_0_tensor_data -> {[31], 6, [16, 17, 8, 15 | List.duplicate(8, 28)]}
        :with_q5_1_tensor_data -> {[32], 7, [16, 17, 8, 15 | List.duplicate(8, 28)]}
        :with_unaligned_q5_1_tensor_data -> {[31], 7, [16, 17, 8, 15 | List.duplicate(8, 28)]}
        :with_q8_0_tensor_data -> {[32], 8, [0, 2, -4, 6 | List.duplicate(0, 28)]}
        :with_unaligned_q8_0_tensor_data -> {[31], 8, [0, 2, -4, 6 | List.duplicate(0, 28)]}
        :with_q8_1_tensor_data -> {[32], 9, [0, 2, -4, 6 | List.duplicate(0, 28)]}
        :with_unaligned_q8_1_tensor_data -> {[31], 9, [0, 2, -4, 6 | List.duplicate(0, 28)]}
        :with_q2_k_tensor_data -> {[256], 10, [0, 1, 2, 3 | List.duplicate(1, 252)]}
        :with_unaligned_q2_k_tensor_data -> {[255], 10, [0, 1, 2, 3 | List.duplicate(1, 252)]}
        :with_q3_k_tensor_data -> {[256], 11, [-4, -1, 0, 3 | List.duplicate(0, 252)]}
        :with_unaligned_q3_k_tensor_data -> {[255], 11, [-4, -1, 0, 3 | List.duplicate(0, 252)]}
        :with_q4_k_tensor_data -> {[256], 12, [0, 1, 8, 15 | List.duplicate(8, 252)]}
        :with_unaligned_q4_k_tensor_data -> {[255], 12, [0, 1, 8, 15 | List.duplicate(8, 252)]}
        :with_q5_k_tensor_data -> {[256], 13, [0, 17, 8, 31 | List.duplicate(8, 252)]}
        :with_unaligned_q5_k_tensor_data -> {[255], 13, [0, 17, 8, 31 | List.duplicate(8, 252)]}
        :with_q6_k_tensor_data -> {[256], 14, [32, 33, 16, 63 | List.duplicate(0, 252)]}
        :with_unaligned_q6_k_tensor_data -> {[255], 14, [32, 33, 16, 63 | List.duplicate(0, 252)]}
        :with_q8_k_tensor_data -> {[256], 15, [0, 2, -4, 6 | List.duplicate(0, 252)]}
        :with_unaligned_q8_k_tensor_data -> {[255], 15, [0, 2, -4, 6 | List.duplicate(0, 252)]}
        :with_unsupported_tensor_type -> {[2, 2], 99, []}
        _other -> {[2, 2], 0, [1.0, 0.0, 0.0, 1.0]}
      end

    tensor_infos = tensor_info("token_embd.weight", dimensions, tensor_type, 0)

    without_data = IO.iodata_to_binary([header, metadata, tensor_infos])

    case mode do
      :without_tensor_data -> without_data
      :with_tensor_data -> with_aligned_f32_tensor_data(without_data, values)
      :with_rectangular_tensor_data -> with_aligned_f32_tensor_data(without_data, values)
      :with_f16_tensor_data -> with_aligned_f16_tensor_data(without_data, values)
      :with_q4_0_tensor_data -> with_aligned_q4_0_tensor_data(without_data, values)
      :with_unaligned_q4_0_tensor_data -> with_aligned_q4_0_tensor_data(without_data, values)
      :with_q4_1_tensor_data -> with_aligned_q4_1_tensor_data(without_data, values)
      :with_unaligned_q4_1_tensor_data -> with_aligned_q4_1_tensor_data(without_data, values)
      :with_q5_0_tensor_data -> with_aligned_q5_0_tensor_data(without_data, values)
      :with_unaligned_q5_0_tensor_data -> with_aligned_q5_0_tensor_data(without_data, values)
      :with_q5_1_tensor_data -> with_aligned_q5_1_tensor_data(without_data, values)
      :with_unaligned_q5_1_tensor_data -> with_aligned_q5_1_tensor_data(without_data, values)
      :with_q8_0_tensor_data -> with_aligned_q8_0_tensor_data(without_data, values)
      :with_unaligned_q8_0_tensor_data -> with_aligned_q8_0_tensor_data(without_data, values)
      :with_q8_1_tensor_data -> with_aligned_q8_1_tensor_data(without_data, values)
      :with_unaligned_q8_1_tensor_data -> with_aligned_q8_1_tensor_data(without_data, values)
      :with_q2_k_tensor_data -> with_aligned_q2_k_tensor_data(without_data, values)
      :with_unaligned_q2_k_tensor_data -> with_aligned_q2_k_tensor_data(without_data, values)
      :with_q3_k_tensor_data -> with_aligned_q3_k_tensor_data(without_data, values)
      :with_unaligned_q3_k_tensor_data -> with_aligned_q3_k_tensor_data(without_data, values)
      :with_q4_k_tensor_data -> with_aligned_q4_k_tensor_data(without_data, values)
      :with_unaligned_q4_k_tensor_data -> with_aligned_q4_k_tensor_data(without_data, values)
      :with_q5_k_tensor_data -> with_aligned_q5_k_tensor_data(without_data, values)
      :with_unaligned_q5_k_tensor_data -> with_aligned_q5_k_tensor_data(without_data, values)
      :with_q6_k_tensor_data -> with_aligned_q6_k_tensor_data(without_data, values)
      :with_unaligned_q6_k_tensor_data -> with_aligned_q6_k_tensor_data(without_data, values)
      :with_q8_k_tensor_data -> with_aligned_q8_k_tensor_data(without_data, values)
      :with_unaligned_q8_k_tensor_data -> with_aligned_q8_k_tensor_data(without_data, values)
      :with_unsupported_tensor_type -> without_data
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

  defp with_aligned_q4_0_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    tensor_data = [<<0x3C00::little-unsigned-integer-size(16)>>, q4_0_bytes(values)]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp with_aligned_q4_1_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)

    tensor_data = [
      <<0x3C00::little-unsigned-integer-size(16)>>,
      <<0x4900::little-unsigned-integer-size(16)>>,
      q4_0_bytes(values)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp with_aligned_q5_0_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    {high_bits, low_bits} = q5_bits(values)

    tensor_data = [
      <<0x3C00::little-unsigned-integer-size(16)>>,
      <<high_bits::little-unsigned-integer-size(32)>>,
      q4_0_bytes(low_bits)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp with_aligned_q5_1_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    {high_bits, low_bits} = q5_bits(values)

    tensor_data = [
      <<0x3C00::little-unsigned-integer-size(16)>>,
      <<0x4900::little-unsigned-integer-size(16)>>,
      <<high_bits::little-unsigned-integer-size(32)>>,
      q4_0_bytes(low_bits)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp q5_bits(values) do
    values
    |> Enum.with_index()
    |> Enum.reduce({0, []}, fn {value, index}, {high_bits, low_bits} ->
      high_bits =
        if value >= 16 do
          Bitwise.bor(high_bits, Bitwise.bsl(1, index))
        else
          high_bits
        end

      {high_bits, [Bitwise.band(value, 0x0F) | low_bits]}
    end)
    |> then(fn {high_bits, low_bits} -> {high_bits, Enum.reverse(low_bits)} end)
  end

  defp with_aligned_q8_0_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    tensor_data = [<<0x3800::little-unsigned-integer-size(16)>>, Enum.map(values, &i8/1)]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp with_aligned_q8_1_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)

    tensor_data = [
      <<0x3800::little-unsigned-integer-size(16)>>,
      <<0x3C00::little-unsigned-integer-size(16)>>,
      Enum.map(values, &i8/1)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp with_aligned_q2_k_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)

    tensor_data = [
      List.duplicate(0x21, 16) |> Enum.map(&<<&1>>),
      q2_k_quantized(values),
      <<0x3C00::little-unsigned-integer-size(16)>>,
      <<0x3C00::little-unsigned-integer-size(16)>>
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp q2_k_quantized(values) do
    values
    |> Enum.chunk_every(128)
    |> Enum.flat_map(fn chunk ->
      0..31
      |> Enum.map(fn index ->
        low_pair =
          Enum.at(chunk, rem(index, 16) + if(index < 16, do: 0, else: 16))
          |> Bitwise.band(0x03)

        Enum.reduce(1..3, low_pair, fn pair_index, byte ->
          offset = pair_index * 32 + rem(index, 16) + if(index < 16, do: 0, else: 16)

          Bitwise.bor(
            byte,
            chunk |> Enum.at(offset) |> Bitwise.band(0x03) |> Bitwise.bsl(pair_index * 2)
          )
        end)
      end)
    end)
    |> Enum.map(&<<&1>>)
  end

  defp with_aligned_q3_k_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    {high_mask, quantized} = q3_k_bits(values)

    tensor_data = [
      high_mask,
      quantized,
      q3_k_scales(List.duplicate(33, 16)),
      <<0x3C00::little-unsigned-integer-size(16)>>
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp q3_k_bits(values) do
    values
    |> Enum.with_index()
    |> Enum.reduce({List.duplicate(0, 32), []}, fn {value, index}, {high_mask, low_bits} ->
      encoded = value + 4
      high_index = rem(index, 32)
      high_bit = Bitwise.bsl(1, div(index, 32))

      high_mask =
        if encoded > 3 do
          List.update_at(high_mask, high_index, &Bitwise.bor(&1, high_bit))
        else
          high_mask
        end

      {high_mask, [Bitwise.band(encoded, 0x03) | low_bits]}
    end)
    |> then(fn {high_mask, low_bits} ->
      {Enum.map(high_mask, &<<&1>>), q2_k_quantized(Enum.reverse(low_bits))}
    end)
  end

  defp q3_k_scales(scales) do
    low =
      0..7
      |> Enum.map(fn index ->
        first = Enum.at(scales, index)
        second = Enum.at(scales, index + 8)

        Bitwise.bor(Bitwise.band(first, 0x0F), Bitwise.bsl(Bitwise.band(second, 0x0F), 4))
      end)

    high =
      0..3
      |> Enum.map(fn index ->
        Enum.reduce(0..3, 0, fn group, byte ->
          scale_index = index + group * 4
          high_bits = Enum.at(scales, scale_index) |> Bitwise.bsr(4) |> Bitwise.band(0x03)

          Bitwise.bor(byte, Bitwise.bsl(high_bits, group * 2))
        end)
      end)

    Enum.map(low ++ high, &<<&1>>)
  end

  defp with_aligned_q4_k_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)

    tensor_data = [
      <<0x3C00::little-unsigned-integer-size(16)>>,
      <<0x3C00::little-unsigned-integer-size(16)>>,
      q4_k_scales(List.duplicate(1, 8), List.duplicate(2, 8)),
      q4_k_quantized(values)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp q4_k_scales(scales, minimums) do
    first =
      0..3
      |> Enum.map(fn index ->
        scale = Enum.at(scales, index)
        high_scale = Enum.at(scales, index + 4) |> Bitwise.bsr(4)

        scale |> Bitwise.band(0x3F) |> Bitwise.bor(Bitwise.bsl(high_scale, 6))
      end)

    second =
      0..3
      |> Enum.map(fn index ->
        minimum = Enum.at(minimums, index)
        high_minimum = Enum.at(minimums, index + 4) |> Bitwise.bsr(4)

        minimum |> Bitwise.band(0x3F) |> Bitwise.bor(Bitwise.bsl(high_minimum, 6))
      end)

    third =
      4..7
      |> Enum.map(fn index ->
        scale = Enum.at(scales, index)
        minimum = Enum.at(minimums, index)

        Bitwise.bor(Bitwise.band(scale, 0x0F), Bitwise.bsl(Bitwise.band(minimum, 0x0F), 4))
      end)

    Enum.map(first ++ second ++ third, &<<&1>>)
  end

  defp q4_k_quantized(values) do
    values
    |> Enum.chunk_every(64)
    |> Enum.flat_map(fn chunk ->
      0..31
      |> Enum.map(fn index ->
        low = chunk |> Enum.at(index) |> Bitwise.band(0x0F)
        high = chunk |> Enum.at(index + 32) |> Bitwise.band(0x0F)

        Bitwise.bor(low, Bitwise.bsl(high, 4))
      end)
    end)
    |> Enum.map(&<<&1>>)
  end

  defp with_aligned_q5_k_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    {high_bits, low_bits} = q5_k_bits(values)

    tensor_data = [
      <<0x3C00::little-unsigned-integer-size(16)>>,
      <<0x3C00::little-unsigned-integer-size(16)>>,
      q4_k_scales(List.duplicate(1, 8), List.duplicate(2, 8)),
      high_bits,
      q4_k_quantized(low_bits)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp q5_k_bits(values) do
    values
    |> Enum.with_index()
    |> Enum.reduce({List.duplicate(0, 32), []}, fn {value, index}, {high_bytes, low_bits} ->
      group_index = div(index, 64)
      group_offset = rem(index, 64)
      high_index = rem(group_offset, 32)
      high_mask = Bitwise.bsl(if(group_offset < 32, do: 1, else: 2), group_index * 2)

      high_bytes =
        if value >= 16 do
          List.update_at(high_bytes, high_index, &Bitwise.bor(&1, high_mask))
        else
          high_bytes
        end

      {high_bytes, [Bitwise.band(value, 0x0F) | low_bits]}
    end)
    |> then(fn {high_bytes, low_bits} ->
      {Enum.map(high_bytes, &<<&1>>), Enum.reverse(low_bits)}
    end)
  end

  defp with_aligned_q6_k_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)
    {high_bits, low_bits} = q6_k_bits(values)

    tensor_data = [
      q4_0_bytes(low_bits),
      high_bits,
      List.duplicate(1, 16) |> Enum.map(&i8/1),
      <<0x3C00::little-unsigned-integer-size(16)>>
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp q6_k_bits(values) do
    values
    |> Enum.with_index()
    |> Enum.reduce({List.duplicate(0, 64), []}, fn {value, index}, {high_bytes, low_bits} ->
      high_value = value |> Bitwise.bsr(4) |> Bitwise.band(0x03)
      high_index = div(index, 4)
      shift = rem(index, 4) * 2

      high_bytes =
        List.update_at(high_bytes, high_index, fn byte ->
          Bitwise.bor(byte, Bitwise.bsl(high_value, shift))
        end)

      {high_bytes, [Bitwise.band(value, 0x0F) | low_bits]}
    end)
    |> then(fn {high_bytes, low_bits} ->
      {Enum.map(high_bytes, &<<&1>>), Enum.reverse(low_bits)}
    end)
  end

  defp with_aligned_q8_k_tensor_data(binary, values) do
    padding = rem(32 - rem(byte_size(binary), 32), 32)

    tensor_data = [
      <<1.0::little-float-size(32)>>,
      Enum.map(values, &i8/1),
      :binary.copy(<<0>>, 32)
    ]

    IO.iodata_to_binary([binary, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp tiny_gguf_with_output_tensors do
    tiny_multi_tensor_gguf(
      block_count: 0,
      tensors: [
        {"token_embd.weight", [2, 2], [1.0, 0.0, 0.0, 1.0]},
        {"output_norm.weight", [2], [1.0, 1.0]},
        {"output.weight", [2, 2], [1.0, 0.0, 0.0, 1.0]}
      ]
    )
  end

  defp tiny_gguf_with_special_token_tensors do
    tiny_multi_tensor_gguf(
      block_count: 0,
      tokens: ["<unk>", "<s>", "</s>", "hello"],
      extra_metadata: special_token_metadata(),
      tensors: [
        {"token_embd.weight", [2, 4], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0]}
      ]
    )
  end

  defp tiny_gguf_with_transformer_tensors do
    identity = [1.0, 0.0, 0.0, 1.0]

    tiny_multi_tensor_gguf(
      block_count: 1,
      tensors: [
        {"token_embd.weight", [2, 2], identity},
        {"blk.0.attn_norm.weight", [2], [1.0, 1.0]},
        {"blk.0.attn_q.weight", [2, 2], identity},
        {"blk.0.attn_k.weight", [2, 2], identity},
        {"blk.0.attn_v.weight", [2, 2], identity},
        {"blk.0.attn_output.weight", [2, 2], identity},
        {"output_norm.weight", [2], [1.0, 1.0]},
        {"output.weight", [2, 2], identity}
      ]
    )
  end

  defp tiny_gguf_with_feed_forward_tensors do
    identity = [1.0, 0.0, 0.0, 1.0]

    tiny_multi_tensor_gguf(
      block_count: 1,
      feed_forward_size: 2,
      tensors: [
        {"token_embd.weight", [2, 2], identity},
        {"blk.0.attn_norm.weight", [2], [1.0, 1.0]},
        {"blk.0.attn_q.weight", [2, 2], identity},
        {"blk.0.attn_k.weight", [2, 2], identity},
        {"blk.0.attn_v.weight", [2, 2], identity},
        {"blk.0.attn_output.weight", [2, 2], identity},
        {"blk.0.ffn_norm.weight", [2], [1.0, 1.0]},
        {"blk.0.ffn_gate.weight", [2, 2], identity},
        {"blk.0.ffn_up.weight", [2, 2], identity},
        {"blk.0.ffn_down.weight", [2, 2], identity},
        {"output_norm.weight", [2], [1.0, 1.0]},
        {"output.weight", [2, 2], identity}
      ]
    )
  end

  defp tiny_multi_tensor_gguf(opts) do
    tensors = Keyword.fetch!(opts, :tensors)
    block_count = Keyword.fetch!(opts, :block_count)
    feed_forward_size = Keyword.get(opts, :feed_forward_size, 8)
    tokens = Keyword.get(opts, :tokens, ["<unk>", "hello"])
    extra_metadata = Keyword.get(opts, :extra_metadata, [])

    metadata =
      [
        kv_string("general.architecture", "llama"),
        kv_u32("general.alignment", 32),
        kv_u32("llama.embedding_length", 2),
        kv_u32("llama.context_length", 16),
        kv_u32("llama.block_count", block_count),
        kv_u32("llama.attention.head_count", 2),
        kv_u32("llama.attention.head_count_kv", 1),
        kv_u32("llama.feed_forward_length", feed_forward_size),
        kv_array_string("tokenizer.ggml.tokens", tokens)
      ] ++ extra_metadata

    header = [
      "GGUF",
      u32(3),
      u64(length(tensors)),
      u64(length(metadata))
    ]

    {tensor_infos, tensor_data} = f32_tensor_sections(tensors)

    without_data = IO.iodata_to_binary([header, metadata, tensor_infos])
    padding = rem(32 - rem(byte_size(without_data), 32), 32)

    IO.iodata_to_binary([without_data, :binary.copy(<<0>>, padding), tensor_data])
  end

  defp tensor_info(name, dimensions, tensor_type, offset) do
    [
      gguf_string(name),
      u32(length(dimensions)),
      Enum.map(dimensions, &u64/1),
      u32(tensor_type),
      u64(offset)
    ]
  end

  defp f32_values(values) do
    values
    |> Enum.map(fn value -> <<value::little-float-size(32)>> end)
    |> IO.iodata_to_binary()
  end

  defp f32_tensor_sections(tensors) do
    {infos, data, _offset} =
      Enum.reduce(tensors, {[], [], 0}, fn {name, dimensions, values}, {infos, data, offset} ->
        binary = f32_values(values)
        padding = rem(32 - rem(byte_size(binary), 32), 32)

        {
          [tensor_info(name, dimensions, 0, offset) | infos],
          [data, binary, :binary.copy(<<0>>, padding)],
          offset + byte_size(binary) + padding
        }
      end)

    {Enum.reverse(infos), data}
  end

  defp q4_0_bytes(values) do
    values
    |> Enum.chunk_every(2)
    |> Enum.map(fn [low, high] -> <<Bitwise.bor(low, Bitwise.bsl(high, 4))>> end)
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

  defp tiny_special_token_gguf do
    metadata =
      [
        kv_string("general.architecture", "llama"),
        kv_array_string("tokenizer.ggml.tokens", ["<unk>", "<s>", "</s>", "hello"])
      ] ++ special_token_metadata()

    header = [
      "GGUF",
      u32(3),
      u64(0),
      u64(length(metadata))
    ]

    IO.iodata_to_binary([header, metadata])
  end

  defp tiny_chat_template_gguf do
    metadata = [
      kv_string("general.architecture", "llama"),
      kv_array_string("tokenizer.ggml.tokens", ["<unk>", "Hello"]),
      kv_string("tokenizer.chat_template", chatml_template())
    ]

    header = [
      "GGUF",
      u32(3),
      u64(0),
      u64(length(metadata))
    ]

    IO.iodata_to_binary([header, metadata])
  end

  defp tiny_usable_chat_template_gguf do
    metadata = [
      kv_string("general.architecture", "llama"),
      kv_array_string("tokenizer.ggml.tokens", ["<unk>", "<|im_start|>", "<|im_end|>"]),
      kv_string("tokenizer.chat_template", chatml_template())
    ]

    header = [
      "GGUF",
      u32(3),
      u64(0),
      u64(length(metadata))
    ]

    IO.iodata_to_binary([header, metadata])
  end

  defp tiny_byte_token_gguf do
    metadata = [
      kv_string("general.architecture", "llama"),
      kv_array_string("tokenizer.ggml.tokens", ["<unk>", "<0x68>", "<0x69>"]),
      kv_u32("tokenizer.ggml.unknown_token_id", 0),
      kv_array_u32("tokenizer.ggml.token_type", [2, 6, 6])
    ]

    header = [
      "GGUF",
      u32(3),
      u64(0),
      u64(length(metadata))
    ]

    IO.iodata_to_binary([header, metadata])
  end

  defp special_token_metadata do
    [
      kv_u32("tokenizer.ggml.unknown_token_id", 0),
      kv_u32("tokenizer.ggml.bos_token_id", 1),
      kv_u32("tokenizer.ggml.eos_token_id", 2),
      kv_bool("tokenizer.ggml.add_bos_token", true),
      kv_bool("tokenizer.ggml.add_eos_token", false),
      kv_array_u32("tokenizer.ggml.token_type", [2, 3, 3, 1])
    ]
  end

  defp chatml_template do
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  end

  defp role_marker_template do
    "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"
  end

  defp kv_string(key, value), do: [gguf_string(key), u32(8), gguf_string(value)]
  defp kv_u32(key, value), do: [gguf_string(key), u32(4), u32(value)]
  defp kv_bool(key, value), do: [gguf_string(key), u32(7), if(value, do: <<1>>, else: <<0>>)]

  defp kv_array_u32(key, values),
    do: [gguf_string(key), u32(9), u32(4), u64(length(values)), Enum.map(values, &u32/1)]

  defp kv_array_string(key, values) do
    [gguf_string(key), u32(9), u32(8), u64(length(values)), Enum.map(values, &gguf_string/1)]
  end

  defp gguf_string(value) do
    [u64(byte_size(value)), value]
  end

  defp u32(value), do: <<value::little-unsigned-integer-size(32)>>
  defp u64(value), do: <<value::little-unsigned-integer-size(64)>>
  defp i8(value), do: <<value::signed-integer-size(8)>>
end
