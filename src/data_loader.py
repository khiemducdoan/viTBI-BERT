import os
import pathlib
import torch
import pandas as pd
import transformers

from torch.utils.data import Dataset, DataLoader
import numpy as np 
import random
import json
from random import shuffle
from mtranslate import translate
import re
import seaborn as sns
class DataAugmentation:
    def __init__(self, stop_words_file, wordnet_file, seed=1):
        self.seed = seed
        random.seed(self.seed)
        
        self.stop_words = self.load_stop_words(stop_words_file)
        self.wordnet_data = self.load_wordnet(wordnet_file)
    
    @staticmethod
    def load_stop_words(file_path):
        stop_words = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                stop_words.append(line.strip())
        return stop_words
    
    @staticmethod
    def load_wordnet(file_path):
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("WordNet file not found.")
            return {}
    
    @staticmethod
    def back_translation(sentence, intermediate_langs=['en', 'fr', 'ru']):
        intermediate_lang = random.choice(intermediate_langs)
        translated_sentence = translate(sentence, intermediate_lang)
        back_translated_sentence = translate(translated_sentence, 'vi')
        return back_translated_sentence
    
    def get_synonyms(self, word):
        synonyms = set()
        for key, value in self.wordnet_data.items():
            if key.strip() == word:
                synonyms.update([v.strip() for v in value])
        synonyms.discard(word)  # Remove the word itself if present
        return list(synonyms)
    
    @staticmethod
    def random_deletion(words, p):
        if len(words) == 1:
            return words
        new_words = [word for word in words if random.uniform(0, 1) > p]
        return new_words if new_words else [random.choice(words)]
    
    @staticmethod
    def random_swap(words, n):
        for _ in range(n):
            if len(words) > 1:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return words
    
    def random_insertion(self, words, n):
        for _ in range(n):
            synonyms = []
            while not synonyms:
                random_word = random.choice(words)
                synonyms = self.get_synonyms(random_word)
            random_synonym = random.choice(synonyms)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_synonym)
        return words
    
    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = [word for word in words if word not in self.stop_words]
        shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
                if num_replaced >= n:
                    break
        return new_words
    
    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, bt_langues=['en', 'fr', 'ru']):
        words = sentence.split(' ')
        num_words = len(words)
        augmented_sentences = [sentence]
    
        if num_words > 1:
            num_new_per_technique = int(num_aug / 5) + 1  # Updated to account for back_translation too
            n_sr = max(1, int(alpha_sr * num_words))
            n_ri = max(1, int(alpha_ri * num_words))
            n_rs = max(1, int(alpha_rs * num_words))
    
            for _ in range(num_new_per_technique):
                augmented_sentences.append(' '.join(self.synonym_replacement(words, n_sr)))
            for _ in range(num_new_per_technique):
                augmented_sentences.append(' '.join(self.random_insertion(words, n_ri)))
            for _ in range(num_new_per_technique):
                augmented_sentences.append(' '.join(self.random_swap(words, n_rs)))
            for _ in range(num_new_per_technique):
                augmented_sentences.append(' '.join(self.random_deletion(words, p_rd)))
            for _ in range(num_new_per_technique):
                augmented_sentences.append(self.back_translation(sentence, intermediate_langs=bt_langues))
        
        return list(set(augmented_sentences))
    
    def edafor3(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, bt_langues=['en', 'fr', 'ru']):
        words = sentence.split(' ')
        num_words = len(words)
        augmented_sentences = [sentence]
    
        if num_words > 1:
            num_new_per_technique = int(num_aug / 5) + 1  # Updated to account for back_translation too
            n_sr = max(1, int(alpha_sr * num_words))
            n_ri = max(1, int(alpha_ri * num_words))
            n_rs = max(1, int(alpha_rs * num_words))
    
            augmented_sentences.append(self.back_translation(sentence, intermediate_langs=bt_langues))
        
        return list(set(augmented_sentences))
    
    @staticmethod
    def clear_punctuation(sentence):
        return re.sub(r'[^\w\s]', '', sentence)
    
    def augment_dataframe(self, df, num_aug=9, alpha=0.1, max_aug_for_3=40):
        augmented_rows = []
        augment_count_3 = 0
        
        for _, row in df.iterrows():
            original_col1 = row[df.columns[0]].strip()
            label = str(row[df.columns[1]]).strip()
    
            if label == "1.0" or label == "4.0":
                augmented_col1 = self.eda(original_col1, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
                for sent1 in augmented_col1:
                    augmented_rows.append({df.columns[0]: self.clear_punctuation(sent1), df.columns[1]: float(label)})
            elif label == "3.0" and augment_count_3 < max_aug_for_3:
                augmented_col1 = self.edafor3(original_col1, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=1)
                for sent1 in augmented_col1:
                    augmented_rows.append({df.columns[0]: self.clear_punctuation(sent1), df.columns[1]: float(label)})
                augment_count_3 += 1
            else:
                augmented_rows.append(row.to_dict())
    
        return pd.DataFrame(augmented_rows)

class ViTBERT(Dataset):
    def __init__(self, data_path,stop_words_file, wordnet_file,indices, type , tokenizer=None):
        self.data_augmentor = DataAugmentation(stop_words_file, wordnet_file)
        if indices is None:    
            self.dataset = pd.read_csv(data_path)
        else: 
            self.dataset = pd.read_csv(data_path).iloc[indices]

        #only get dataset in index, index is taken from train ids by kfold from sklearn
        if type == "train":
            self.dataset = self.data_augmentor.augment_dataframe(self.dataset, num_aug=2, alpha=0.1, max_aug_for_3=40)
        else:
            self.dataset = self.dataset
        #if type = "train", augment dataset 
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    def __getitem__(self, idx):
        text = self.dataset.iloc[idx, 0]  # Extract text from dataframe
        label = self.dataset.iloc[idx, 1]-1.0  # Extract label

        # Tokenization
        tokenized_text = self.tokenizer(text, add_special_tokens=True, max_length=100, padding='max_length', truncation=True, return_tensors='pt')

        # Extract fields from tokenized text
        input_ids = tokenized_text['input_ids'].squeeze()
        attention_mask = tokenized_text['attention_mask'].squeeze()

        # Convert label to tensor
        # label = torch.tensor(label, dtype=torch.long)  # Changed dtype to torch.long

        # Return tensors
        return input_ids, attention_mask, label, text  # Return attention_mask

    def __len__(self):
        return len(self.dataset)

def main():
    data_path = "/media/data3/home/khiemdd/ViTBERT/dataset/donedataset_train.csv"
    tokenizer = "demdecuong/vihealthbert-base-word"
    
    dataset = ViTBERT(data_path, tokenizer=tokenizer)
    test_loader  = DataLoader(dataset, batch_size=3)
    input_ids, attention_mask, label, text = next(iter(test_loader))
    print("Input IDs 1:", input_ids)
    print("Attention Mask 1:", attention_mask)
    print("Label 1: ", label)
    print("Labels shape1: ", label.shape)
    print("Input IDs Shape 1:", input_ids.shape)
    input_ids, attention_mask, label, text = dataset[0]  # Accessing the first item
    
    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)
    print("Label:", label)
    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)
    print("text: ", text )

if __name__ == "__main__":
    main()
