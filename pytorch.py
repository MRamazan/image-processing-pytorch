import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
import os
import cv2 as cv


# GoogleNet modelini yükleyin
googlenet = models.googlenet(pretrained=True)

# Modelin son sınıf katmanını değiştirin (ImageNet için 1000 sınıf yerine)
#.fc fully connected anlamına gelir googlenet imagenet verisinini bi katmanıdır. bu katman 1000 sınıf içerir biz bunu 24 ile değiştiriyoz
#.nn ifadesi yapay sinir ağı mimarisinde değişiklik yapmak için kullanılır burda fully connected katmanında değişiklik yapıldı
num_classes = 24  # Örnek olarak, 10 farklı sınıf için
#in_features girdi özellik sayısı , out features çıktı özellik sayısı. burda yerine num classes konulmuş. nn.linear bi sınıftır parametrelerini değiştiriyoz. .fc de katmandır.
googlenet.fc = nn.Linear(googlenet.fc.in_features, num_classes)

# Eğer GPU kullanıyorsanız, modeli GPU üzerinde çalışacak şekilde taşıyın
device = torch.device("cpu")
# Bu satır modeli Cihaza taşır. GPU ya veya CPU ya. ve o cihazda çalışmasını sağlar.
googlenet = googlenet.to(device)
print(device)

# Modeli eğitim için kullanmak üzere ayarlayın
criterion = nn.CrossEntropyLoss()  # Kayıp fonksiyonudur. bu fonksiyon sistemin çıktısı ile doğru çıktı arasındaki kaybı belirler ve sistemin daha iyi tahmin yapması için sinyal gönderir.
optimizer = torch.optim.SGD(googlenet.parameters(), lr=0.001, momentum=0.9)  # Optimizasyon algoritmasıdır. ağırlıkları günceller. lr her güncelleme adımında ağırlıkların ne kadar değiştirilceiğini kontrol eder. momentum ağırlık güncellemesi yapılırken gradyan yönünü ve hızını ayarlar.


arc_yol = r"C:\Users\PC\PycharmProjects\pythonProject\architectural-styles-dataset" # dataset'in  yolu


data = []# eğitim verileri
label = [] # test verileri

#Test verilerini etiketleme yeniden boyutlandırma ve listeye ekleme
def labeling(yol, etiket):
 for x in os.listdir(yol):
    tamyol = os.path.join(yol, x)
    resim = cv.imread(tamyol)
    print(tamyol)
    resized = cv.resize(resim, (196, 196))

    if resized is not None:
          data.append(resized)
          label.append(etiket)



# test klasöründeki dosyaların liste halinde oluşturulması
arc = os.listdir(arc_yol)

# test klasörünün yolu

# kaçıncı döngü tespiti ve arc listesindeki elemanların for döngüsü
for x, y in enumerate(arc):
     tamyoll = os.path.join(arc_yol, y)
     labeling(tamyoll, x)


from torch.utils.data import Dataset

#dataseti oluşturmak için sınıf
class CustomDataset(Dataset):
    def __init__(self, data, etiket, transform=None):
        self.image_list = data
        self.label_list = etiket
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)
        #çağırıldığında resim ve etiket değerini döndürür
        return image, label

from torchvision import transforms

# Veri normalizasyonu
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Veri dönüşümünüzü tanımlarken normalizasyonu da ekleyin
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


# test klasörünün yolu
test_yol = (r"C:\Users\PC\PycharmProjects\pythonProject\arctest")

# test klasörü içindeki klasörlerin isimlerinin listelenmiş hali
arctestt = os.listdir(test_yol)

test_data = []  # Test verilerini depolamak için boş bir liste oluşturun
test_labels = []  # Test etiketlerini depolamak için boş bir liste oluşturun


# test verilerini işlemek ve etiketlemek için fonksiyon
def testpre(etikett):
        sinif = arctestt[etikett]
        yoll = os.path.join(test_yol, sinif)
        sinif_testleri = os.listdir(yoll)
        for x in sinif_testleri:
            resim_yolu = os.path.join(yoll, x)
            c = cv.imread(resim_yolu)
            resized = cv.resize(c, (196, 196))

            if resized is not None:
                test_data.append(resized)
                test_labels.append(etikett)
#fonksiyonu 24 kere çalıştırır bu sayede 24 klasörede işlem uygulanır. ilk sınıf 0 etiketiyle başlar son sınıf 23 etiketiyle biter
for x in range(0, 24):
    testpre(x)






from torch.utils.data import DataLoader

batch_size = 128  # Her toplu işlemde kaç veri örneği kullanılacağını belirler
train_dataset = CustomDataset(data, label, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# eğitim verilerinin modele gönderilmesi ve ağırlıkların ayarlanmasını
for epoch in range(25):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = googlenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# test verisi için DataLoader oluşturun
test_dataset = CustomDataset(test_data, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = googlenet
model.eval()  # modeli değerlendirme moduna alın
correct = 0
total = 0


with torch.no_grad():  # Gradyan hesaplama yapmadan geçiş yapın
    # test verilerinin eğittiğimiz modele gönderilmesi ve sınıflandırma yapılması
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += inputs.size(0)
        correct += (predicted == labels).sum().item()

    # doğruluk oranı
    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')






