from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

class_name = ["mainland China politics", "Hong Kong - Macau politics", "International news", "financial news",
              "culture", "entertainment", "sports"]
path = 'model/roberta-base-finetuned-chinanews-chinese'
model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

test_text = '《007》首周票房2.15亿 ，挤走《一代宗师》，丹尼尔·克雷格第三次出演007。'
test_token = tokenizer(test_text, return_tensors="pt").data['input_ids']
result_logits = model(test_token).logits
probs = torch.nn.functional.softmax(result_logits, dim=-1)
predicted_class = torch.argmax(probs).item()
print('prediction is：' + class_name[predicted_class])
