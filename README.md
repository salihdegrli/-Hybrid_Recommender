# Hybrid_Recommender(item Based & user Based recommender)

## İş Problemi
ID'si verilen kullanıcı için item-based ve 
user-based recommender yöntemlerini 
kullanarak tahmin yapınız.

## Veri Seti Hikayesi
- Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır.
- İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını 
barındırmaktadır.
- 27.278 filmde 2.000.0263 derecelendirme içermektedir. 
- Bu veriler 138.493 kullanıcı tarafından 09 Ocak 1995 ile 31 Mart 2015
tarihleri arasında oluşturulmuştur. Bu veri seti ise 17 Ekim 2016 tarihinde 
oluşturulmuştur.
- Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy 
verdiği bilgisi mevcuttur.

## Değişkenler

- movie.csv
  - movieId – Eşsiz film numarası. (UniqueID)
  - title – Film adı

- rating.csv
  - userid – Eşsiz kullanıcı numarası. (UniqueID)
  - movieId – Eşsiz film numarası. (UniqueID)
  - rating – Kullanıcı tarafından filme verilen puan
  - timestamp – Değerlendirme tarihi
