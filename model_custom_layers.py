""" Doc-string """

import os
import torch
import warnings
import contextlib
from torch import nn
from torch import autograd
from torch.nn import functional as F

enabled = True
weight_gradients_disabled = False
conv2d_gradfix_cache = dict()

@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled

    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8."]) and input.device.type != "cpu" and enabled and torch.backends.cudnn.enabled:
        implemented_function = conv2d_gradfix(transpose=False,
                                              weight_shape=weight.shape,
                                              stride=stride,
                                              padding=padding,
                                              output_padding=0,
                                              dilation=dilation,
                                              groups=groups).apply(input, weight, bias)
    else:
        implemented_function = F.conv2d(input=input,
                                        weight=weight,
                                        bias=bias,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)
    return implemented_function


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8."]) and input.device.type != "cpu" and enabled and torch.backends.cudnn.enabled:
        implemented_function = conv2d_gradfix(transpose=True,
                                              weight_shape=weight.shape,
                                              stride=stride,
                                              padding=padding,
                                              output_padding=output_padding,
                                              groups=groups,
                                              dilation=dilation).apply(input, weight, bias)
    else:
        implemented_function = F.conv_transpose2d(input=input,
                                                  weight=weight,
                                                  bias=bias,
                                                  stride=stride,
                                                  padding=padding,
                                                  output_padding=output_padding,
                                                  dilation=dilation,
                                                  groups=groups)
    return implemented_function


def conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):

    def ensure_tuple(xs, ndim):
        xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
        return xs

    class Conv2d(autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            if transpose:
                out = F.conv_transpose2d(input=input,
                                         weight=weight,
                                         bias=bias,
                                         output_padding=output_padding,
                                         **common_kwargs)
            else:
                out = F.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

            ctx.save_for_backward(input, weight)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = None, None, None

            if ctx.needs_input_grad[0]:
                if transpose:
                    p = [0, 0]
                else:
                    p = [input.shape[i + 2] - (grad_output.shape[i + 2] - 1) * stride[i] - (1 - 2 * padding[i]) - dilation[i] * (weight_shape[i + 2] - 1) for i in range(ndim)]

                grad_input = conv2d_gradfix(transpose=(not transpose),
                                            weight_shape=weight_shape,
                                            output_padding=p,
                                            **common_kwargs).apply(grad_output, weight, None)

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))

            return grad_input, grad_weight, grad_bias

    class Conv2dGradWeight(autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation("aten::cudnn_convolution_transpose_backward_weight" if transpose else "aten::cudnn_convolution_backward_weight")
            flags = [torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32]
            grad_weight = op(weight_shape,
                             grad_output,
                             input,
                             padding,
                             stride,
                             dilation,
                             groups,
                             *flags)
            ctx.save_for_backward(grad_output, input)

            return grad_weight

        @staticmethod
        def backward(ctx, grad_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_grad_output, grad_grad_input = None, None

            if ctx.needs_input_grad[0]:
                grad_grad_output = Conv2d.apply(input, grad_grad_weight, None)

            if ctx.needs_input_grad[1]:
                if transpose:
                    p = [0, 0]
                else:
                    p = [input.shape[i + 2] - (grad_output.shape[i + 2] - 1) * stride[i] - (1 - 2 * padding[i]) - dilation[i] * (weight_shape[i + 2] - 1) for i in range(ndim)]
                grad_grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs).apply(grad_output, grad_grad_weight, None)

            return grad_grad_output, grad_grad_input

    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = ensure_tuple(stride, ndim)
    padding = ensure_tuple(padding, ndim)
    output_padding = ensure_tuple(output_padding, ndim)
    dilation = ensure_tuple(dilation, ndim)

    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in conv2d_gradfix_cache:
        return conv2d_gradfix_cache[key]

    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    conv2d_gradfix_cache[key] = Conv2d
    return Conv2d


class FusedLeakyReLU(nn.Module):
    """ Doc-string """
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        """ Doc-string """
        # pylint: disable=no-member
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input_tensor):
        """ Doc-string """
        rest_dim = [1] * (input_tensor.ndim - self.bias.ndim - 1)
        shaped_bias = self.bias.view(1, *rest_dim, self.bias.shape[0]) if input_tensor.ndim == 3 else self.bias.view(1, self.bias.shape[0], *rest_dim)
        out = F.leaky_relu(input_tensor + shaped_bias, negative_slope=self.negative_slope) * self.scale
        return out


    class FusedLeakyReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, bias, negative_slope, scale):
            ctx.bias = bias is not None

            empty = input.new_empty(0)
            bias = empty if bias is None else bias

            out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
            ctx.save_for_backward(out)
            ctx.negative_slope = negative_slope
            ctx.scale = scale

            return out

        @staticmethod
        def backward(ctx, grad_output):
            out, = ctx.saved_tensors

            grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale)
            grad_bias = None if not ctx.bias else grad_bias

            return grad_input, grad_bias, None, None

    class FusedLeakyReLUFunctionBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, out, bias, negative_slope, scale):
            ctx.save_for_backward(out)
            ctx.negative_slope = negative_slope
            ctx.scale = scale

            empty = grad_output.new_empty(0)
            grad_input = fused.fused_bias_act(grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale)

            dim = [0]
            if grad_input.ndim > 2:
                dim += list(range(2, grad_input.ndim))

            grad_bias = grad_input.sum(dim).detach() if bias else empty

            return grad_input, grad_bias

        @staticmethod
        def backward(ctx, gradgrad_input, gradgrad_bias):
            out, = ctx.saved_tensors
            gradgrad_out = fused.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)

            return gradgrad_out, None, None, None, None


def fused_leaky_relu(input_tensor, bias, negative_slope=0.2, scale=2**0.5):
    """ Doc-string """
    if bias is not None:
        rest_dim = [1] * (input_tensor.ndim - bias.ndim - 1)
        shaped_bias = bias.view(1, *rest_dim, bias.shape[0]) if input_tensor.ndim == 3 else bias.view(1, bias.shape[0], *rest_dim)
        out = F.leaky_relu(input_tensor + shaped_bias, negative_slope=negative_slope) * scale
    else:
        out = F.leaky_relu(input_tensor, negative_slope=negative_slope) * scale
    # FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    return out


class UpFirDn2d(nn.Module):
    """ Doc-string """
    def __init__(self, kernel, up=1, down=1, pad=(0, 0)):
        """ Doc-string """
        # pylint: disable=no-member
        super().__init__()

        self.data = dict(kernel=kernel,
                         flipped_kernel=torch.flip(kernel, [0, 1]),
                         up=(up, up),
                         down=(down, down),
                         pad=(pad[0], pad[1], pad[0], pad[1]))

    def forward(self, input_tensor):
        """ Doc-string """
        kernel_h, kernel_w = self.data['kernel'].shape
        batch, channel, in_h, in_w = input_tensor.shape
        pad_x0, pad_x1, pad_y0, pad_y1 = self.data['pad']
        up_x, up_y = self.data['up'][0], self.data['up'][1]
        down_x, down_y = self.data['down'][0], self.data['down'][1]
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

        self.data['grad_in_shape'] = (-1, in_h, in_w, 1)
        self.data['grad_out_shape'] = (-1, out_h, out_w, 1)
        self.data['in_shape'] = (batch, channel, in_h, in_w)
        self.data['out_shape'] = (batch, channel, out_h, out_w)
        self.data['g_pad'] = (kernel_w - pad_x0 - 1,
                              kernel_h - pad_y0 - 1,
                              in_w * up_x - out_w * down_x + pad_x0 - up_x + 1,
                              in_h * up_y - out_h * down_y + pad_y0 - up_y + 1)

        out = UpFirDn2dForward.apply(input_tensor, self.data)
        return out

    def upfirdn2d_native(input_tensor, kernel, up, down, pad):
        """ Doc-string """
        # pylint: disable=no-member
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        _, in_h, in_w, channel = input_tensor.shape

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        convolution_weight = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)

        input_tensor = input_tensor.view(-1, in_h, 1, in_w, 1, 1)
        input_tensor = F.pad(input_tensor, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
        input_tensor = input_tensor.view(-1, in_h * up_y, in_w * up_x, 1)
        padded_input = F.pad(input_tensor, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])

        clipping = [max(-pad_y0, 0),
                    padded_input.shape[1] - max(-pad_y1, 0),
                    max(-pad_x0, 0),
                    padded_input.shape[2] - max(-pad_x1, 0)]
        clipped_input = padded_input[:, clipping[0]:clipping[1], clipping[2]:clipping[3], :].permute(0, 3, 1, 2)

        clipped_input = clipped_input.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
        out = F.conv2d(clipped_input, convolution_weight)
        out = out.reshape(-1, 1, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)

        out = out.permute(0, 2, 3, 1)[:, ::down_y, ::down_x, :]
        return out.view(-1, channel, out_h, out_w)

    class UpFirDn2dForward(torch.autograd.Function):
        """ Doc-string """
        @staticmethod
        def forward(ctx, input_tensor, data):
            """ Doc-string """
            input_tensor = input_tensor.reshape(*data['grad_in_shape'])
            out = upfirdn2d_native(input_tensor, data['kernel'], data['up'], data['down'], data['pad']).view(*data['out_shape'])

            ctx.save_for_backward(data['kernel'], data['flipped_kernel'])
            ctx.data = data

            return out

        @staticmethod
        def backward(ctx, grad_output):
            """ Doc-string """
            kernel, grad_kernel = ctx.saved_tensors

            grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.data)
            return grad_input, None, None, None, None

    class UpFirDn2dBackward(torch.autograd.Function):
        """ Doc-string """
        @staticmethod
        def forward(ctx, grad_output, kernel, grad_kernel, data):
            """ Doc-string """
            grad_output = grad_output.reshape(*data['grad_out_shape'])
            grad_input = upfirdn2d_native(grad_output, grad_kernel, data['up'], data['down'], data['g_pad']).view(*data['in_shape'])

            ctx.save_for_backward(kernel)
            ctx.data = data

            return grad_input

        @staticmethod
        def backward(ctx, gradgrad_input):
            """ Doc-string """
            kernel, = ctx.saved_tensors

            gradgrad_input = gradgrad_input.reshape(*ctx.data['in_shape'])
            gradgrad_out = upfirdn2d_native(gradgrad_input, kernel, ctx.data['up'], ctx.data['down'], ctx.data['pad']).view(*ctx.data['out_shape'])

            return gradgrad_out, None, None, None, None, None, None, None, None


def upfirdn2d(input_tensor, kernel, up=1, down=1, pad=(0, 0)):
    """ Doc-string """
    # pylint: disable=no-member
    kernel_h, kernel_w = kernel.shape
    batch, channel, in_h, in_w = input_tensor.shape
    out_h = (in_h * up + pad[0] + pad[1] - kernel_h) // down + 1
    out_w = (in_w * up + pad[0] + pad[1] - kernel_w) // down + 1

    data = dict(kernel=kernel,
                flipped_kernel=torch.flip(kernel, [0, 1]),
                up=(up, up),
                down=(down, down),
                in_shape=(-1, channel, in_h, in_w),
                out_shape=(-1, channel, out_h, out_w),
                pad=(pad[0], pad[1], pad[0], pad[1]),
                grad_in_shape=(-1, in_h, in_w, 1),
                grad_out_shape=(-1, out_h, out_w, 1),
                g_pad=(kernel_w - pad[0] - 1,
                       kernel_h - pad[0] - 1,
                       in_w * up - out_w * down + pad[0] - up + 1,
                       in_h * up - out_h * down + pad[0] - up + 1))

    out = UpFirDn2dForward.apply(input_tensor, data)
    return out
