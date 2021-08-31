from net import alexnet, resnet20
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T
from tqdm import tqdm


def main():
    transform_train = T.Compose([
                           T.RandomCrop(32, padding=4),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                          )

    transform_val = T.Compose([
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                          )

    train_set = CIFAR10(root='./data', train=True, transform=transform_train, download=False)
    val_set = CIFAR10(root='./data', train=False, transform=transform_val, download=False)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, pin_memory=True)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    kwarg = {'w_bits': 1, 'a_bits': 2, 'num_classes': 10}
    # model = alexnet(**kwarg)
    model = resnet20(**kwarg)
    print(model)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 150, 180], gamma=0.1)

    EPOCHS = 200
    for EPOCH in range(EPOCHS):
        model.train()
        train_loss = 0.

        with tqdm(train_loader) as tbar:
            for i, data in enumerate(tbar):
                optim.zero_grad()

                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, targets)
                train_loss += loss
                tbar.set_description(
                    f'EPOCH: {EPOCH} || total_train_loss: {train_loss / (i+1):3f} batch_train_loss: {loss:.3f}')
                loss.backward()
                optim.step()
            scheduler.step(EPOCH)

        model.eval()
        val_loss = 0.
        total = 0
        correct = 0
        with tqdm(val_loader) as tbar:
            for i, data in enumerate(tbar):

                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                with torch.no_grad():
                    outputs = model(imgs)

                loss = criterion(outputs, targets)
                val_loss += loss
                tbar.set_description(
                    f'EPOCH: {EPOCH} || total_train_loss: {val_loss / (i + 1):3f} batch_train_loss: {loss:.3f}')
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f'\n Accuracy of the network on the 10000 test images: {round(100 * correct / total, 3)}% '
              f'|| Curruent Learning rate: {scheduler.get_lr()[0]}')

    state_dict = {'model_state_dict': model.state_dict()}
    torch.save(state_dict, './model.pt')


if __name__ == '__main__':
    main()