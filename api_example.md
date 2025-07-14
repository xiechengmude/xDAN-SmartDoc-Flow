curl -X POST "https://api.siliconflow.cn/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "将图片中的内容用列表的形式总结\n\n## 字段处理细则\n\n- 必须识别出勾选框的勾选状态，统一用：[√]表示选中，[ ]表示未选，属于同一组的勾选框使用一行输出\n\n## 特别注意\n\n- 注意识别图片中的手写体文字，不要忽略\n- 必须将所有的内容都提取出来，注意根据语义保留层级关系，不能遗漏任何信息！！！\n- 不需要输出任何其他字眼，只需要用Markdown列表格式输出提取后的内容！！！"
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "识别这张图片的内容，使用Markdown格式输出，注意保留层级关系！！！"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,YOUR_BASE64_ENCODED_IMAGE"
            }
          }
        ]
      }
    ],
    "max_tokens": 4000,
    "temperature": 0.1
  }'