#!/usr/bin/env bash

MODELS=(
# openai/gpt-5-nano
# anthropic/claude-4.5-sonnet-thinking-on-10k
# anthropic/claude-4.5-sonnet-thinking-off
# xai/grok-4-fast-thinking-off
# xai/grok-4-fast-thinking-on
# togetherai/moonshotai/Kimi-K2-Instruct-0905
# google/gemini-2.5-flash-thinking-off
# google/gemini-2.5-flash-thinking-on
# google/gemini-2.5-pro-thinking-on
# togetherai/openai/gpt-oss-20b
# togetherai/openai/gpt-oss-120b
# deepseek-ai/DeepSeek-V3.1-Terminus-thinking-off
# deepseek-ai/DeepSeek-V3.2-Exp-thinking-off
deepseek-ai/DeepSeek-V3.2-Exp-thinking-on
deepseek-ai/DeepSeek-V3.1-Terminus-thinking-on
)

for model in "${MODELS[@]}"; do
  echo "Running evaluation for ${model}"
  python evaluation.py --model_name "${model}" --proc_num 100 
done
