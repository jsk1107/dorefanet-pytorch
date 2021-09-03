# Pytorch를 활용한 DoReFa-Net 구현

양자화 학습을 위해서 DoReFa-Net을 직접 구현해보았습니다.

구현을 위해서 [zzzxxxttt](https://github.com/zzzxxxttt/pytorch_DoReFaNet),
[mohdumar644](https://github.com/mohdumar644/DoReFaNet-PyTorch)
저장소를 참고하였습니다.

논문에서 제시한 실험 환경과, 성능지표를 달성하기 위해서 셋팅을 하였으며,
Gradient를 양자화 하는 부분은 개발하지 않았습니다.

가중치와, 활성함수를 K-bits로 양자화 할 수 있으며, 데이터 셋으로는 Cifar-10을 사용하였습니다.

## Requirements:
- python==3.7
- pytorch==1.9.0+cu111
- torchvision==0.9.0

## CIFAR-10:

업데이트 예정.