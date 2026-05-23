curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model":"Qwen/Qwen3-4B-Instruct-2507",
  "messages":[
    {"role":"user","content":"介绍一下Transformer"}
  ]
}'