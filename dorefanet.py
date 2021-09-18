# -*- coding: utf-8 -*-
# Original author: jsk1107

import torch
import torch.nn.functional as F
from torch import Tensor # Typing 하기 위해 import 시켰음.


class Quantizer(torch.nn.Module):
    r"""
    양자화가 진행되기 위한 Helper 클래스. 실제 작동은 Quantize 에서 실행된다.

    Quantize 클래스틑 torch.autograd.Function 을 상속 받기 때문에, apply 함수를 통해 데이터를 넘겨줄 수 있다.

    Parameters
    ----------
    k: int
        양자화 비트
    """
    def __init__(self, k: int, name: str) -> None:
        super(Quantizer, self).__init__()
        self.bits = k
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        # if self.name == 'weight' and (torch.min(x) < -1 or torch.max(x) > 1):
        #     raise ValueError('Weight는 양자화 되기 전, 입력값이 -1 <= x <= 1 사이여야 합니다.')
        # if self.name == 'activation' and torch.min(x) < 0 or torch.max(x) > 1:
        #     raise ValueError('Activation은 양자화 되기 전, 입력값이 0 <= x <= 1 사이여야 합니다.')
        return Quantize.apply(x, self.bits, self.name)


class Quantize(torch.autograd.Function):
    r"""
    실제 양자화가 일어나는 클래스.

    DoReFa-Net에서는 Forward에는 Uniform 형태의 양자화를 진행하고, Backward에서는 STE를 사용한다.

    넘어온 기울기을 그대로 앞으로 전달한다.

    .. math::

        forward: w_{q} = \frac{1}{2^{k}-1} \times round(w_{i} \times (2^{k}-1))

    .. math::

        backward: \frac{\partial c}{\partial w_{i}} = \frac{\partial c}{\partial w_{q}}

        (즉, grad_in = grad_out)
    """

    @staticmethod
    def forward(ctx: object, x: Tensor, k: int, name: str) -> Tensor:
        r"""
        k-bits 양자화 변환.

        Parameters
        ----------
        ctx: object
            데이터 입출력 인스턴스. 모든 순전파, 역전파 정보를 저장하거나 로드할 수 있다.
        x: Tensor
            입력 데이터. 양자화 되기 전 값이다.
        k: int
            bits number.

        Returns
        -------
        r_o: Tensor
            k-bits로 양자화가 완료된 값.
        """

        # if name == 'weight' and (torch.min(r_o) < -1 or torch.max(r_o) > 1):
        #     raise ValueError('Weight는 양자화 되기 전, 입력값이 -1 <= x <= 1 사이여야 합니다.')
        # if name == 'activation' and torch.min(r_o) < 0 or torch.max(r_o) > 1:
        #     raise ValueError('Activation은 양자화 되기 전, 입력값이 0 <= x <= 1 사이여야 합니다.')

        if name == 'weight' and k == 1:
            r_o = torch.sign(x)
            r_o[r_o == 0] == 1
        else:
            q = 2 ** k - 1
            r_o = torch.round(q * x) / q

        return r_o

    @staticmethod
    def backward(ctx: object, grad_out: Tensor) -> Tensor:
        r"""
        역전파 과정. 순전파를 할때 bits number를 같이 받아줬기 때문에 코드 상으로는 해당 영역으로도 역전파가 되어 나가야한다.
        하지만, 실제로 bits number에는 역전파가 전달되면 안되기 때문에, None을 리턴해준다.

        Parameters
        ----------
        ctx: object
            데이터 입출력 인스턴스. 모든 순전파, 역전파 정보를 저장하거나 로드할 수 있다.

        grad_out: Tensor
            역전파되어 건너온 기울기. STE를 사용하므로 그대로 앞쪽으로 전파해준다.

        Returns
        -------
        grad_input: Tensor
            앞쪽 레이어로 전파되어 나가는 기울기.
        """

        grad_input = grad_out.clone()
        return grad_input, None, None


class QuantizationWeight(torch.nn.Module):
    r"""
    Weight를 양자화 하기 위한 클래스.

    Quantizer는 호출해서 생성자로 만들어두기 때문에, Network를 설계할때는 QuantizationWeight 클래스만을 사용한다.

    Paramerters
    -----------
    w_bits: int
        weight bits number.

    Attributes
    ----------
    quantizer: Quantizer
        양자화 Helper 클래스.
    """

    def __init__(self, w_bits: int) -> None:
        super(QuantizationWeight, self).__init__()
        self.bits = w_bits
        self.quantizer = Quantizer(self.bits, name='weight')

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 진행시, bits에 따라 양자화가 다르게 발생한다.

        32-bit 일때는 양자화를 진행하지 않는다.

        1-bit 일때는 signum 함수를 사용한다. 이때 주의할점은 torch.sign = {-1, 0, 1}을 리턴한다(Ternary).
        따라서 0을 1 또는 -1로 치환하여 사용한다. 여기서는 1로 치환함.

        2~31bit 일때는 tanh를 이용한 수식을 사용한다.

        Reference:
            Paper 참조(3~4page): "https://arxiv.org/pdf/1606.06160.pdf"

        """
        if self.bits == 1:
            mu = torch.mean(torch.abs(x)).detach() # mu는 상수이기 때문에 계산그래프에서 제외해야함.
            w_q = self.quantizer(x / mu) * mu
        elif self.bits == 32:
            w_q = x
        else:
            # FIXME: torch.tanh(x)에 대해서 역전파가 수행되는거 같다.
            #  여기를 detach() 하면 계산그래프에서 떨어지기 때문에 Parameter Update가 안됨.
            #  x의 계산그래프를 가지고 가면서 torch.tanh를 할 수 있는 방법을 찾아야함.
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach() # max_w는 상수이기 때문에 계산그래프에서 제외해야함.
            r_i = (weight / (2 * max_w)) + 0.5
            w_q = 2 * self.quantizer(r_i) - 1
        return w_q


class QuantizationActivation(torch.nn.Module):
    """
    Activation을 양자화 하기 위한 클래스.

    ReLU까지 수행한 후, 해당 인스턴스를 호출하여 사용하여야 한다.

    Examples
    --------
    >>> quant_act = QuantizationActivation(a_bits=2)
    >>> relu = nn.ReLU(inplace=True)
    >>> ...
    >>> # forward 영역에서
    >>> ...
    >>> x = relu(x)
    >>> x = quant_act(x)
    """
    def __init__(self, a_bits: int) -> None:
        super(QuantizationActivation, self).__init__()
        self.bits = a_bits
        self.quantizer = Quantizer(self.bits, name='activation')

    def forward(self, x: Tensor) -> Tensor:
        if self.bits == 32:
            a_q = x
        else:
            a_q = self.quantizer(torch.clamp(x, 0, 1))
        return a_q


class QuantizationConv2d(torch.nn.Conv2d):
    """
    Conv2d 클래스를 상속받아 오버라이딩한다. 양자화를 진행하고 Convolution을 진행.
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int=1,
        padding: int=0,
        dilation: int=1,
        groups: int=1,
        bias: bool=True,
        w_bits: int=1
    ) -> None:
        super(QuantizationConv2d, self).__init__(
            in_channel, out_channel, kernel_size, stride, padding, dilation, groups,bias)
        self.quantized_weight = QuantizationWeight(w_bits)

    def forward(self, x: Tensor) -> Tensor:
        w_q = self.quantized_weight(self.weight)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizationFullyConnected(torch.nn.Linear):
    """
    Linear 클래스를 상속받아 오버라이딩한다. 양자화를 진행하고 Linear를 진행.
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        bias: bool=True,
        w_bits: int=1
    ) -> None:
        super(QuantizationFullyConnected, self).__init__(in_channel, out_channel, bias)
        self.quantized_weight = QuantizationWeight(w_bits)

    def forward(self, x: Tensor) -> Tensor:
        w_q = self.quantized_weight(self.weight)
        return F.linear(x, w_q, self.bias)
