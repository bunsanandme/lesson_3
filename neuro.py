# Ипротируем библиотеки для работы 
import torchvision
import torchvision.transforms as transforms 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


# Загружаем FashionMNIST в папку "dataset"
train_set = torchvision.datasets.FashionMNIST(root = "dataset/", train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = "dataset/", train = False, download = True, transform = transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False)
torch.manual_seed(0)



# Создадим архитектуру сверточной сети в виде класса (стиль ООП)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Функции активации
        self.activ_func = torch.relu 
        #self.activ_func = torch.tanh
        self.activ_func = torch.sigmoid
        #self.activ_func = nn.functional.elu

        # 1 канал входного изображения, 32 выходных канала, квадратное ядро свертки 5x5
        self.conv1 = nn.Conv2d(1, 32, 5)

        # 32 канала входного изображения, 64 выходных канала, квадратное ядро свертки 5x5
        self.conv2 = nn.Conv2d(32, 64, 5)

        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)  # 4*4 from image dimension
        self.fc2 = nn.Linear(1024, 256)

        # Выходной слой
        self.output = nn.Linear(256, 10)

        # Реализация Dropout
        self.dropout = nn.Dropout(p = 1.0)


    # Прямое распространение ошибки
    def forward(self, x):
        # Операции подвыборки с сеткой 2*2
        x = torch.max_pool2d(self.activ_func(self.conv1(x)), (2, 2))
        x = torch.max_pool2d(self.activ_func(self.conv2(x)), (2, 2))

        x = torch.flatten(x, 1) # Сгладим данные кроме размера сетки

        # Передаем данные в полнозвязные слои
        x = self.activ_func(self.fc1(x))
        x = self.activ_func(self.fc2(x)) 
        
        # Активируем дропаут
        x = self.dropout(x)
        
        # Передаем в выходной слои
        x = self.output(x)
        return x


net = Net()
print(net)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= 0.1)

print('Начинаем обучение модели')

num_epochs = 3

train_accuracies = []
train_losses = []

for epoch in range(num_epochs):

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(training_loader, 0):
        # переменная data = [входные данные, метки]
        inputs, labels = data
        optimizer.zero_grad()

        # ПРО + ОРО + оптимизация
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Класс с макс. значение = предсказание модели
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    train_accuracies.append(accuracy)
    train_losses.append(running_loss)

    print('Эпоха [{}]/[{}] Потеря: {} Точность: {} %'.format(epoch + 1,num_epochs, running_loss, accuracy))
        
print('Окончание обучения')

model_scripted = torch.jit.script(net) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save

print('Тестирование модели')

num_epochs = 3

test_accuracies = []
test_losses = []

for epoch in range(num_epochs): 

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(training_loader, 0):
        images, labels = data
        # Вычисляем выходные данные через нейросеть
        outputs = net(images)
        loss = criterion(outputs, labels)
       
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
    
    accuracy = 100 * correct / total

    test_accuracies.append(accuracy)
    test_losses.append(running_loss)

    print('Эпоха [{}]/[{}] Потеря: {} Точность: {} %'.format(epoch + 1,num_epochs, running_loss, accuracy))
print('Окончание тестирования')


# Рисуем графики
plt.plot(range(1,num_epochs+1),train_accuracies,label='Train')
plt.plot(range(1,num_epochs+1),test_accuracies,label='Test')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.savefig('figures/train_test_accuracy_plot.png')
plt.show()

plt.plot(range(1,num_epochs+1),train_losses,label='Train')
plt.plot(range(1,num_epochs+1),test_accuracies,label='Test')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.savefig('figures/train_test_losses_plot.png')
plt.show()