import torch
import pymysql
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForMaskedLM, AdamW
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import re
import os
def clean_text_korean(text):
    
    cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return cleaned_text


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',  # Ensure returns are tensors
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df['text'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=True)

def train_epoch(model, data_loader, optimizer, device, scheduler, data_collator):
    model.train()
    losses = []

    for batch in tqdm(data_loader):
        inputs = {key: value.to(device) for key, value in batch.items()}
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)

def eval_model(model, data_loader, device, data_collator):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = {key: value.to(device) for key, value in batch.items()}
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            losses.append(loss.item())

    return np.mean(losses)

def main():


    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='wodnd1205', db='mat_rnd', charset='utf8')
    cursor = conn.cursor()

    for year in range(2010, 2023):
        query = f"""SELECT `과제고유번호` as doc, concat(`과제명`,`요약문_연구목표`, `요약문_연구내용`, `요약문_기대효과`) as text
                    FROM mat_rnd.ntis_rnd
                    WHERE (`과학기술표준분류1-대` ='에너지/자원' OR `과학기술표준분류2-대` ='에너지/자원' OR `과학기술표준분류3-대` ='에너지/자원')
                    AND `과제명` is not null AND `요약문_연구목표` is not null AND `요약문_연구내용` is not null AND `요약문_기대효과` is not null 
                    AND `계속과제여부` = '신규' AND `제출년도` <= {year};
        """
        cursor.execute(query)
        data = cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ['doc', 'text']
        data['cleaned_text'] = data['text'].apply(clean_text_korean)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
        model = BertForMaskedLM.from_pretrained("skt/kobert-base-v1")
        model = model.to(device)

        RANDOM_SEED = 42
        BATCH_SIZE = 4
        EPOCHS = 3
        LEARNING_RATE = 2e-5
        MAX_LEN = 512
        df_train, df_val = train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)
        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)


        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, data_collator)
            print(f'Train loss {train_loss}')

            val_loss = eval_model(model, val_data_loader, device, data_collator)
            print(f'Val loss {val_loss}')
        
        # Saving the model and tokenizer
        save_directory = f"./plm/kobert_{year}"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        print(f"Model and tokenizer saved to {save_directory}")


    conn.close()



if __name__ == '__main__':
    main()