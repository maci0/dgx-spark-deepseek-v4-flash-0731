# Client integration (OpenAI-compatible harnesses)

This serve is OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, `/v1/completions`), served
model name `deepseek-v4-flash`, streaming + tools supported. One quirk trips up reasoning-aware
clients.

## The `reasoning` vs `reasoning_content` gotcha

This vLLM build (0.21.1rc1 + overlays) returns chain-of-thought under the message field **`reasoning`**.
DeepSeek's **hosted** API returns it under **`reasoning_content`**. Clients hard-wired to
`reasoning_content` won't find the reasoning here, and the model's `</think>` bleeds into the
displayed `content`.

Verify which field your serve uses:

```bash
curl -s http://HEAD_IP:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reason then answer: is 51 prime?"}],
       "max_tokens":200,"chat_template_kwargs":{"thinking":true}}' \
  | python3 -c 'import sys,json;m=json.load(sys.stdin)["choices"][0]["message"];print("field:", "reasoning" if m.get("reasoning") else "reasoning_content" if m.get("reasoning_content") else "none")'
```

## Kimi Code

Point Kimi Code at the serve via an OpenAI-compatible provider and set **`reasoning_key = "reasoning"`**
on the model entry (this is the whole fix — without it, `</think>` leaks into content):

`~/.kimi-code/config.toml`:

```toml
[providers.vllm-local]
base_url = "http://HEAD_IP:8000/v1"
type = "openai"
api_key = "FREE"

[models."vllm-local/deepseek-v4-flash"]
provider = "vllm-local"
model = "deepseek-v4-flash"
max_context_size = 1000000          # not 100000 — or you silently cap the 1M ctx
capabilities = [ "thinking", "tool_use" ]
reasoning_key = "reasoning"         # <-- our field name (Kimi defaults to reasoning_content)
support_efforts = [ "low", "high", "max" ]
```

Kimi matches capabilities by model-name prefix and auto-handles `reasoning_content` for third-party
endpoints — but only pulls reasoning from a **different** field if you name it via `reasoning_key`.

Test headless (no `</think>` in the output = fixed):

```bash
kimi -p "Reason step by step whether 51 is prime, then answer." -m vllm-local/deepseek-v4-flash
```

## Notes for other harnesses (lm-eval, etc.)

- Harnesses that send `stop` sequences can decapitate reasoning mid-`<think>` (`</think>` never
  arrives → `content=null`). The runtime's patch 0005 guards this server-side by scoping stop-strings
  to content. If you see empty-content-with-tokens-billed, that's the class of bug.
- Coding traffic maximizes DSpark acceptance (mean accepted length ~4.3 on code vs ~2.4 on math), so
  agentic coding clients see the best tok/s.
