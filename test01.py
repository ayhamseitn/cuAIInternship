from transformers import Trainer, TrainingArguments
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer, Trainer, TrainingArguments
import torch

# Tokenizer ve Modeli Yükleyin
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdForQuestionAnswering.from_pretrained('google/bigbird-roberta-base')

# Örnek Eğitim ve Doğrulama Verisi
train_texts = ["MADDE 1 – (1)Bu yönergenin amacı, önlisans ve lisans düzeyindeki öğrencilerin Çukurova Üniversitesindeki fakülte, yüksekokul, konservatuvar veya meslek yüksekokulu bünyesinde yer alan diploma programları arasında veya diğer yükseköğretim kurumlarından Çukurova Üniversitesindeki eşdeğer diploma programlarına yatay geçiş usul ve esaslarını düzenlemektir. Tıp ve Diş hekimliği fakültelerinde ilgili fakültelerin kendi yatay geçiş yönergeleri esas alınır.", "MADDE 2 – (1)Bu yönerge, 24.04.2010 tarihli 27561 sayılı Resmi Gazete’de yayımlanan, Yükseköğretim Kurumlarında Ön Lisans ve Lisans Düzeyindeki Programlar Arasında Geçiş, Çift Ana Dal, Yan Dal ile Kurumlar Arası Kredi Transferi Yapılması Esaslarına İlişkin Yönetmelikhükümlerine dayanılarak hazırlanmıştır."]
train_questions = ["Bu yönergenin amacı nedir?", "Bu yönerge hangi yönetmelik hükümlerine dayandırılarak hazırlanmıştır?"]
train_answers = [{"text": "önlisans ve lisans düzeyindeki öğrencilerin Çukurova Üniversitesindeki fakülte, yüksekokul, konservatuvar veya meslek yüksekokulu bünyesinde yer alan diploma programları arasında veya diğer yükseköğretim kurumlarından Çukurova Üniversitesindeki eşdeğer diploma programlarına yatay geçiş usul ve esaslarını düzenlemektir.", "answer_start": 34}, {"text": "24.04.2010 tarihli 27561 sayılı Resmi Gazete’de yayımlanan, Yükseköğretim Kurumlarında Ön Lisans ve Lisans Düzeyindeki Programlar Arasında Geçiş, Çift Ana Dal, Yan Dal ile Kurumlar Arası Kredi Transferi Yapılması Esaslarına İlişkin Yönetmelik hükümlerine dayanılarak hazırlanmıştır.", "answer_start": 25}]

val_texts = ["MADDE 4 – (1)Farklı yükseköğretim kurumlarının diploma programları veya Çukurova Üniversitesindeki diploma programları arasında, ancak önceden ilan edilen sayı ve geçiş şartları çerçevesinde geçiş yapılabilir.", "MADDE 5–(1) (Değişik - Senato-18/04/2017-07/06) Önlisans ve lisans diploma programlarının hazırlık sınıfına; önlisans diploma programlarının ilk yarıyılı ile son yarıyılına, lisans diploma programlarının ilk iki yarıyılı ile son iki yarıyılına yatay geçiş yapılamaz."]
val_questions = ["Çukurova Üniversitesi'ndeki diploma programları arasında geçiş nasıl yapılır?", "Hangi durumlarda yatay geçiş yapılamaz?"]
val_answers = [{"text": "ancak önceden ilan edilen sayı ve geçiş şartları çerçevesinde geçiş yapılabilir.", "answer_start": 129}, {"text": "Önlisans ve lisans diploma programlarının hazırlık sınıfına; önlisans diploma programlarının ilk yarıyılı ile son yarıyılına, lisans diploma programlarının ilk iki yarıyılı ile son iki yarıyılına yatay geçiş yapılamaz.", "answer_start": 48}]

# Tokenizasyon
train_encodings = tokenizer(train_texts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, val_questions, truncation=True, padding=True)

# Dataset Sınıfı
class SQuADDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, answers):
        self.encodings = encodings
        self.answers = answers

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['start_positions'] = torch.tensor(self.answers[idx]['answer_start'])
        item['end_positions'] = torch.tensor(self.answers[idx]['answer_start'] + len(self.answers[idx]['text']))
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SQuADDataset(train_encodings, train_answers)
val_dataset = SQuADDataset(val_encodings, val_answers)

# Eğitim Parametreleri
training_args = TrainingArguments(
    output_dir='C:\\Users\\Dell\\Desktop\\Staj\\cuAI01\\Model002\\results',           # Modelin ve konfigürasyonun kaydedileceği dizin
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Eğitim
trainer.train()

# Modeli Kaydetme
trainer.save_model('C:\\Users\\Dell\\Desktop\\Staj\\cuAI01\\Model002\\results')
