from net import alexnet, resnet20
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T
from tqdm import tqdm
from logger import get_logger


logger = get_logger('./log', log_config='./logging.json')

def main(**kwarg):
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

    # 실험 조건
    model = resnet20(**kwarg)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 150, 180], gamma=0.1)
    EPOCHS = 200

    logger.info(f'W_bits: {kwarg.get("w_bits")} | A_bits: {kwarg.get("a_bits")}')

    best_acc = 0.
    for EPOCH in range(EPOCHS):
        model.train()
        train_loss = 0.

        with tqdm(train_loader) as tbar:
            for i, data in enumerate(tbar):
                optim.zero_grad()

                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                outputs = model(imgs)
                print(outputs)
                print(outputs.shape)
                print(targets)
                print(targets.shape)
                loss = criterion(outputs, targets)
                train_loss += loss
                tbar.set_description(
                    f'EPOCH: {EPOCH + 1} | total_train_loss: {train_loss / (i+1):.4f} | batch_train_loss: {loss:.4f}')
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
                    f'EPOCH: {EPOCH + 1} | val_train_loss: {val_loss / (i + 1):.4f} | batch_val_loss: {loss:.4f}')
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = 100 * correct / total
        logger.info(f'EPOCH: {EPOCH + 1} | '
                    f'Loss: {val_loss / (i + 1):.4f} | '
                    f'Accuracy: {acc:.4f}%'
                    )

        if best_acc < acc:
            best_acc = acc
            state_dict = {'model_state_dict': model.state_dict(),
                          'ECPOH': EPOCH}
            torch.save(state_dict, f'./model_w_{w_bits}_a{a_bits}.pt')
    logger.info(f'Best_Acc: {best_acc:.4f}%')

if __name__ == '__main__':
    w_a = [[2, 32], [4, 4], [4, 32], [32, 32]]
    for cfg in w_a:
        w_bits, a_bits = cfg
        kwarg = {'w_bits': w_bits, 'a_bits': a_bits}
        main(**kwarg)