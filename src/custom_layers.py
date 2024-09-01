import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import scipy # type: ignore
class CustomLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        # weight shape - output x input dimension
        # bias shape - output dimension

        # implement y = x (mult) w_transpose + b

        # YOUR IMPLEMENTATION HERE!
        # output=1 is a placeholder
        # print(input.shape, weight.shape, bias.shape)
        if input.dim() == 1:
            input = input.unsqueeze(0) 
        output = torch.mm(input, weight.T) + bias # ??? is this it lol he gave us the formula

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        # save the tensors required for back pass
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Shapes.
        # grad_output - batch x output_count
        # grad_input  - batch x input
        # grad_weight - output x input 
        # grad_bias   - output shape

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # YOUR IMPLEMENTATION HERE!
        t = time.time()

        # y = u * w_T + b
        # dy/du = w
        grad_input = torch.mm(grad_output, weight)
        # print("grad_input:", time.time() - t)
        # for each output y_j = sum(u_i * w.T_(j, i))
        # => dy_j/dw_(a, b) = u_b when a = j, 0 otherwise

        grad_weight = torch.einsum('bi,bj->ij', grad_output, input)
        grad_bias = torch.einsum('ij,j->j', grad_output, torch.ones_like(bias))
            
        # use either print or logger to print its outputs.
        # make sure you disable before submitting
        # print(grad_input)
        # logger.info("grad_output: %s", grad_bias.shape)

        # print('lienar shapes')
        # print(grad_output.shape, grad_input.shape, grad_weight.shape, grad_bias.shape)
        # print("linear backprop:", time.time() - t)

        return grad_input, grad_weight, grad_bias
    
class CustomReLULayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # YOUR IMPLEMENTATION HERE!
        # print(input.shape)
        output = torch.maximum(input, torch.zeros_like(input))
        # print(output.shape)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        t = time.time()

        grad_input = grad_output.clone()
        # YOUR IMPLEMENTATION HERE!
        # dReLU/dx = 1 if x > 0, 0 otherwise (heaviside)
        grad_input[input < 0] = 0 # squeeze and unsqueeze because grad_output is [10], but input is [1, 10] (nvm its not needed anymore but leaving this comment here in case it decides to be funky again)
        # print("relu backprop:", time.time() - t)
        return grad_input


class CustomSoftmaxLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

        # YOUR IMPLEMENTATION HERE!
        softmax_output = torch.nn.Softmax(dim=dim)(input)

        ctx.save_for_backward(softmax_output)
        ctx.dim = dim

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors
        dim = ctx.dim

        # YOUR IMPLEMENTATION HERE!
        t = time.time()

        # https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
        SM = softmax_output.unsqueeze(2)  # Shape: (batch_size, num_classes, 1)
        # print("dshdui")
        # print(SM.shape)
        grad_input = torch.diag_embed(softmax_output) - torch.bmm(SM, SM.transpose(1, 2))
        # print(grad_input.shape, grad_output.shape)
        # grad_input = torch.einsum('bi,bij->bj', grad_output, grad_input)
        grad_input = torch.bmm(grad_input, grad_output.unsqueeze(2)).squeeze(2)

        # print("softmax backprop:", time.time() - t)

        # print(grad_input.shape)
        return grad_input, None

class CustomConvLayer(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias, stride, kernel_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # implement the cross correlation filter
        
        # weight shape - out_ch x in_ch x kernel_width x kernel_height
        # bias shape - out_ch
        # input shape - batch x ch x width x height
        # out shape - batch x out_ch x width //stride x height //stride
        
        # You can assume the following,
        #  no padding
        #  kernel width == kernel height
        #  stride is identical along both axes

        out_ch = weight.shape[0]
        in_ch = weight.shape[1]

        kernel = kernel_size
        
        batch, _, height, width = input.shape

        # YOUR IMPLEMENTATION HERE!
        # output = torch.zeros(batch, out_ch, width // stride, height // stride)
        # for i in range(batch):
        #     for j in range(out_ch):

        # return output
        return 1
    
    @staticmethod
    def cross_correlate(input, kernel, stride):
        out = np.zeros((input.shape[0] // stride, input.shape[1] // stride))
        for i in range(1, input.shape[0] - 1, stride):
            for j in range(1, input.shape[1] - 1, stride):
                filtered_image[i-1, j-1] = np.sum(img[i-1:i+2, j-1:j+2] * kernel)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, stride, kernel_size = inputs
        # save the tensors required for back pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride

    @staticmethod
    def backward(ctx, grad_output):
        # grad output shape - batch x out_dim x out_width x out_height (strided)
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        grad_input = grad_weight = grad_bias = None

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        out_ch = weight.shape[0]
        in_ch = weight.shape[1]

        kernel = weight.shape[2]
        
        batch, _, height, width = input.shape

        # YOUR IMPLEMENTATION HERE!

        return grad_input, grad_weight, grad_bias, None, None