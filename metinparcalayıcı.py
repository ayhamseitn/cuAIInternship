from transformers import BertTokenizer

# Tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")

def split_text(text, max_length=511):
    """Metni anlamlı parçalara böler."""
    # Metni cümlelere ayır
    sentences = text.split('. ')  # Burada cümle ayırıcı olarak '. ' kullanıyoruz

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Cümleyi tokenize et
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(tokenized_sentence)

        if current_length + sentence_length + 1 > max_length:  # +1 for space between sentences
            # Çunkü tamamla ve yeni bir tane başlat
            chunks.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(current_chunk)))
            current_chunk = tokenized_sentence
            current_length = sentence_length
        else:
            # Cümleyi mevcut parçaya ekle
            current_chunk.extend(tokenized_sentence)
            current_length += sentence_length

    if current_chunk:
        # Son parçayı ekle
        chunks.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(current_chunk)))

    return chunks

def main():
    text = "MADDE 5–(1) (Değişik - Senato-18/04/2017-07/06) Önlisans ve lisans diploma programlarının hazırlık sınıfına; önlisans diploma programlarının ilk yarıyılı ile son yarıyılına, lisans diploma programlarının ilk iki yarıyılı ile son iki yarıyılına yatay geçiş yapılamaz. (2) Üniversite içinde aynı diploma programlarında birinci öğretimden ikinci öğretime kontenjan sınırlaması olmaksızın yatay geçiş yapılabilir. Ancak, ikinci öğretim diploma programına geçiş yapan öğrenciler ikinci öğretim ücreti öderler. (3) Birinci öğretim diploma programındaki öğrenciler birinci ya da ikinci öğretim diploma programına yatay geçiş yapabilmek için her iki programa da birbirinden bağımsız olarak başvurabilirler. Her iki diploma programına birden başvuran öğrenci, her iki programa da yatay geçiş yapma şartlarını sağlaması durumunda aksi yönde beyanı bulunmadığı sürece öncelikle birinci öğretim diploma programına yerleştirilir ve diğer programdaki başvurusu değerlendirmeye alınmaz. İkinci öğretimden sadece ikinci öğretim diploma programlarına yatay geçiş yapılabilir. Ancak, ikinci öğretim diploma programlarından başarı bakımından bulunduğu sınıfın ilk yüzde onuna girerek bir üst sınıfa geçen öğrenciler birinci öğretim diploma programlarına kontenjan dâhilinde yatay geçiş yapabilirler. (4) Uzaktan öğretimden, üniversitemiz uzaktan öğretim diploma programlarına yatay geçiş yapılabilir. Uzaktan öğretimden örgün öğretim programlarına geçiş yapılabilmesi için, öğrencinin öğrenim görmekte olduğu programdaki genel not ortalamasının 100 üzerinden 80 veya üzeri olması veya kayıt olduğu yıldaki merkezi yerleştirme puanının, geçmek istediği diploma programının o yılki taban puanına eşit veya yüksek olması gerekir. (5) Birinci veya ikinci öğretim diploma programlarından uzaktan eğitim veren diploma programlarına yatay geçiş yapılabilir."
    
    # Metni parçalara ayır
    chunks = split_text(text, max_length=511)
    
    # Parçaları yazdır
    for i, chunk in enumerate(chunks):
        print(f"Parça {i+1}: {chunk}")
        print()

if __name__ == "__main__":
    main()
