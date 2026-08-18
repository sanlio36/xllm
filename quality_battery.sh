#!/bin/bash
# MTP 质量基线测试:3 个复杂问题 × 单发 + 2 并发,输出是否正常(无循环/无乱码/答案合理)
SVC=$1          # baseline | mtp
OUT=/tmp/qa_${SVC}
mkdir -p $OUT
ask() { # ask <prompt-file> <tag> <concurrent>
  local p="$1" tag="$2" conc="$3"
  if [ "$conc" = "1" ]; then
    curl -s --max-time 300 -H "Content-type: application/json" -X POST -d "{\"model\":\"glm5\",\"stream\":false,\"max_tokens\":300,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}]}" http://127.0.0.1:18994/v1/chat/completions -o $OUT/${tag}.json
  else
    (curl -s --max-time 300 -H "Content-type: application/json" -X POST -d "{\"model\":\"glm5\",\"stream\":false,\"max_tokens\":300,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}]}" http://127.0.0.1:18994/v1/chat/completions -o $OUT/${tag}_a.json &)
    sleep 2
    curl -s --max-time 300 -H "Content-type: application/json" -X POST -d "{\"model\":\"glm5\",\"stream\":false,\"max_tokens\":300,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}]}" http://127.0.0.1:18994/v1/chat/completions -o $OUT/${tag}_b.json
  fi
}
ask "计算123乘以5311,给出完整过程" math 1
ask "写一段100字介绍量子计算的基本原理" quantum 1
ask "从上海到北京有多少种交通方式?分别大约需要多久?" travel 1
ask "计算123乘以5311,给出完整过程" mathc 2
ask "写一段100字介绍量子计算的基本原理" quantumc 2
echo "done -> $OUT"
