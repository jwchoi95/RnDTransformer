import torch
from transformers import AutoTokenizer, BertModel
from sklearn.neighbors import LocalOutlierFactor
import pymysql
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import re

def clean_text_korean(text):
    
    cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return cleaned_text
def embed_text_single(tokenizer, model, text):
    if text is None or len(text.strip()) == 0:
        return None
    
    # Tokenize and encode the text
    inputs = tokenizer.batch_encode_plus([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model output using no gradient calculation
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Get the pooler_output as embedding
    embedding = outputs.pooler_output.squeeze().cpu().numpy()
    return embedding

# Load tokenizer and model once outside the loop

#tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
#model = BertModel.from_pretrained("skt/kobert-base-v1").to(device)

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='wodnd1205', db='mat_rnd', charset='utf8')
cursor = conn.cursor()

for year in range(2022, 2023):
    query = f"""SELECT `과제고유번호` as doc, `과제명` as title, `요약문_연구목표` as aim, `요약문_연구내용` as content, `요약문_기대효과` as expect
                FROM mat_rnd.ntis_rnd
                WHERE (`과학기술표준분류1-대` ='에너지/자원' OR `과학기술표준분류2-대` ='에너지/자원' OR `과학기술표준분류3-대` ='에너지/자원')
                AND `과제명` is not null AND `요약문_연구목표` is not null AND `요약문_연구내용` is not null AND `요약문_기대효과` is not null 
                AND `계속과제여부` = '신규' AND `제출년도` <= {year};
    """
    cursor.execute(query)
    data = cursor.fetchall()
    load_directory = f"./plm/kobert_{year}"
    model = BertModel.from_pretrained(load_directory).to(device)
    tokenizer = AutoTokenizer.from_pretrained(load_directory)
    # Initialize lists for embeddings
    docs = []
    title_embeddings = []
    aim_embeddings = []
    content_embeddings = []
    expect_embeddings = []

    # Process each item individually
    for item in data:
        doc, title, aim, content, expect = item
        title = clean_text_korean(title)
        aim = clean_text_korean(aim)
        content = clean_text_korean(content)
        expect = clean_text_korean(expect)
        
        docs.append(doc)
        
        title_embedding = embed_text_single(tokenizer, model, title)
        if title_embedding is not None:
            title_embeddings.append(title_embedding)
        else:
            print(doc)

    
        aim_embedding = embed_text_single(tokenizer, model, aim)
        if aim_embedding is not None:
            aim_embeddings.append(aim_embedding)
    
    
        content_embedding = embed_text_single(tokenizer, model, content)
        if content_embedding is not None:
            content_embeddings.append(content_embedding)
    
    
        expect_embedding = embed_text_single(tokenizer, model, expect)
        if expect_embedding is not None:
            expect_embeddings.append(expect_embedding)

    # Convert lists to torch tensors
    title_embeddings = torch.tensor(title_embeddings)
    aim_embeddings = torch.tensor(aim_embeddings)
    content_embeddings = torch.tensor(content_embeddings)
    expect_embeddings = torch.tensor(expect_embeddings)

    # Apply LOF
    n = round(len(data)*0.01)
    print(n)
    lof = LocalOutlierFactor(n_neighbors=n, contamination='auto')

    if len(title_embeddings) > 0:
        title_lof_scores = lof.fit_predict(title_embeddings).tolist()
        title_scores = lof.negative_outlier_factor_.tolist()
    if len(aim_embeddings) > 0:
        aim_lof_scores = lof.fit_predict(aim_embeddings).tolist()
        aim_scores = lof.negative_outlier_factor_.tolist()
    if len(content_embeddings) > 0:
        content_lof_scores = lof.fit_predict(content_embeddings).tolist()
        content_scores = lof.negative_outlier_factor_.tolist()
    if len(expect_embeddings) > 0:
        expect_lof_scores = lof.fit_predict(expect_embeddings).tolist()
        expect_scores = lof.negative_outlier_factor_.tolist()
    df = pd.DataFrame({'doc_id': docs, 
                       'title_lof': title_lof_scores, 'title_lof_score': title_scores, 
                       'aim_lof': aim_lof_scores, 'aim_lof_score': aim_scores, 
                       'content_lof': content_lof_scores, 'content_lof_score': content_scores, 
                       'expect_lof': expect_lof_scores, 'expect_lof_score': expect_scores})
    columns_to_normalize = ['title_lof_score', 'aim_lof_score', 'content_lof_score', 'expect_lof_score']


    scaler = MinMaxScaler()
    for col in columns_to_normalize:
        df[f'{col}_norm'] = df[col].abs()  
        df[f'{col}_norm'] = scaler.fit_transform(df[[f'{col}_norm']]) 
    norm_columns = [f'{col}_norm' for col in columns_to_normalize]
    df['average_lof_score'] = df[norm_columns].mean(axis=1)
    df.to_csv(f'lof_{year}_kobert_ft.csv', index = False)


conn.close()