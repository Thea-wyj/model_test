import os
import random
import json
import argparse
import pandas as pd
import jieba
from nltk.corpus import wordnet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载同义词库：从文件 `tongyici.txt` 加载本地同义词
def load_local_synonym_dict(tongyici_file):
    synonym_dict = {}
    with open(tongyici_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue

            word1, _, word2 = parts  # 只取词汇1和词汇2，忽略中间的"同义"
            
            # 为每对同义词添加双向映射
            if word1 not in synonym_dict:
                synonym_dict[word1] = set()
            if word2 not in synonym_dict:
                synonym_dict[word2] = set()
            
            # 将word1和word2作为同义词互相添加
            synonym_dict[word1].add(word2)
            synonym_dict[word2].add(word1)

    return synonym_dict

# 加载SST数据集
def load_sst_data(data_dir, num_samples=10):
    data_before = []
    labels_before = []

    # 加载标签文件
    with open(os.path.join(data_dir, 'label.txt'), 'r', encoding='utf-8') as f:
        labels_before = [label.strip() for label in f.readlines()]

    # 加载测试集数据文件
    df = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)
    df = df.sample(frac=1).head(num_samples)  # 随机打乱数据并选择指定数量的数据
    for idx, row in df.iterrows():
        text = row[1]
        label = row[0]
        data_before.append((text, label))

    return data_before, labels_before

# 使用jieba进行中文分词
def jieba_tokenize(text):
    return list(jieba.cut(text))

# 数据增强
def apply_data_augmentation(data, labels, augmentation_methods, local_synonym_dict):
    data_after = []
    labels_after = []
    label_changes = {}

    for text, label in data:
        data_augmented = [(text, label)]  # 添加原始数据
        print(f"原始文本: {text}")  # 打印原始文本

        # 增强方法应用
        if 'random_swap' in augmentation_methods:
            for _ in range(3):
                augmented_text = random_swap(text)
                print(f"随机交换后的文本: {augmented_text}")  # 打印修改后的文本
                data_augmented.append((augmented_text, label))

        if 'random_deletion' in augmentation_methods:
            augmented_text = random_deletion(text)
            print(f"随机删除后的文本: {augmented_text}")  # 打印修改后的文本
            data_augmented.append((augmented_text, label))

        if 'synonym_replacement' in augmentation_methods:
            for _ in range(3):
                augmented_text = synonym_replacement(text, local_synonym_dict)
                print(f"同义词替换后的文本: {augmented_text}")  # 打印修改后的文本
                data_augmented.append((augmented_text, label))

        if 'random_insertion' in augmentation_methods:
            for _ in range(3):
                augmented_text = random_insertion(text, local_synonym_dict)
                print(f"随机插入后的文本: {augmented_text}")  # 打印修改后的文本
                data_augmented.append((augmented_text, label))

        data_after.extend(data_augmented)
        labels_after.extend([label] * len(data_augmented))

    # 计算标签变化率
    for method in augmentation_methods:
        changed_labels = 0
        total_labels = len(labels_after)
        for i in range(total_labels):
            inputs = tokenizer(data_after[i][0], return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax().item()

            if labels_after[i] != labels_before[predicted_label]:
                changed_labels += 1

        change_rate = changed_labels / total_labels
        label_changes[method] = change_rate

    return data_after, labels_after, label_changes

# 随机交换单词
def random_swap(text):
    words = jieba_tokenize(text)  # 使用jieba分词
    if len(words) < 2:
        return text
    random_idx1 = random.randint(0, len(words) - 1)
    random_idx2 = random.randint(0, len(words) - 1)
    words[random_idx1], words[random_idx2] = words[random_idx2], words[random_idx1]
    return ''.join(words)

# 随机删除单词
def random_deletion(text, p=0.1):
    words = jieba_tokenize(text)  # 使用jieba分词
    remaining_words = [word for word in words if random.uniform(0, 1) > p]
    if len(remaining_words) == 0:
        return text
    return ''.join(remaining_words)

# 同义词替换，替换概率设置为1，优先使用本地同义词库
def synonym_replacement(text, local_synonym_dict):
    words = jieba_tokenize(text)  # 使用jieba分词
    new_words = words.copy()
    for i in range(len(words)):
        word = words[i]
        # 首先尝试使用本地同义词库
        if word in local_synonym_dict:
            new_words[i] = random.choice(list(local_synonym_dict[word]))  # 替换成同义词
        # 如果没有找到同义词，则使用WordNet
        elif wordnet.synsets(word):
            word_synonyms = get_synonyms_from_wordnet(word)
            if word_synonyms:
                new_words[i] = random.choice(word_synonyms)
    return ''.join(new_words)

# 随机插入同义词
def random_insertion(text, local_synonym_dict):
    words = jieba_tokenize(text)  # 使用jieba分词
    new_words = words.copy()
    for _ in range(3):  # 在句子中随机插入3个同义词
        random_word = random.choice(words)
        # 优先使用本地同义词库
        if random_word in local_synonym_dict:
            random_synonym = random.choice(list(local_synonym_dict[random_word]))
            random_idx = random.randint(0, len(new_words) - 1)
            new_words.insert(random_idx, random_synonym)
        # 如果本地库没有该词的同义词，则使用WordNet
        elif wordnet.synsets(random_word):
            word_synonyms = get_synonyms_from_wordnet(random_word)
            if word_synonyms:
                random_synonym = random.choice(word_synonyms)
                random_idx = random.randint(0, len(new_words) - 1)
                new_words.insert(random_idx, random_synonym)
    return ''.join(new_words)

# 获取WordNet同义词
def get_synonyms_from_wordnet(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

# 减去随机值
def subtract_random_values(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        random_value = random.uniform(0.10, 0.20)
        new_value = round(value - random_value, 2)
        new_dict[key] = new_value
    return new_dict

# 解析命令行参数
def parseCmdArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='chinanews',
                        help='model path')

    parser.add_argument('--data_path', type=str,
                        default='data',
                        help='data path')

    parser.add_argument('--tongyici_file', type=str, default='tongyici.txt',
                        help='local synonym file')

    parser.add_argument('--methodList', type=str, nargs='+', default=["random_swap",
     "random_deletion", "synonym_replacement", "random_insertion"], help='method namelist')

    # 解析参数
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseCmdArgument()  # 解析命令行参数

    model_path = args.model_path
    tokenizer_path = model_path
    data_dir = args.data_path
    augmentation_methods = args.methodList
    local_synonym_dict = load_local_synonym_dict(args.tongyici_file)  # 加载本地同义词

    # 加载预训练模型和tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 加载SST数据集的测试数据和标签
    data_before, labels_before = load_sst_data(data_dir, num_samples=10)

    # 数据增强
    data_after, labels_after, label_changes = apply_data_augmentation(data_before, labels_before, augmentation_methods, local_synonym_dict)

    # 打印标签变化率
    print("标签变化率:")
    for method, change_rate in label_changes.items():
        print(f"{method}: {change_rate}")

    # 保存标签变化率到JSON文件
    with open("result.json", "w") as f:
        json.dump(label_changes, f)