import openai


# 设置你的API凭证
openai.api_key = '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'

# 发送一个对话请求
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]
)

# 获取生成的文本
message = response['choices'][0]['message']['content']
print(message)