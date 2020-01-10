"""
The principle is simple: we define two distances, one for the content Dc 
and one for the style Ds. Dc measures how different the content is between 
two images while Ds measures how different the style is between two images. 
Then, we take a third image, the input, and transform it to minimize both its 
content-distance with the content-image and its style-distance with the style-image. 
Now we can import the necessary packages and begin the neural transfer.
"""

from __future__ import print_function

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import torchvision.transforms as transforms 
import torchvision.models as models

from PIL import Image 
import matplotlib.pyplot as plt 
import copy 
torch.__version__

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128 

# converts image input to torch tensor
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader("./images/vangogh.png")
content_img = image_loader("./images/faulkner.jpg")

assert style_img.size() == content_img.size() # must be the same size

unloader = transforms.ToPILImage() # bc tensors aren't georgeous per se 
plt.ion() # enable interactive plots

def imshow(tensor, title=None):
    image = tensor.cpu().clone()    # so as not to modify the tensor 
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)    # allow for update 

plt.figure()
#imshow(style_img, title='style_img')

plt.figure()
#imshow(content_img, title='content_img')

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        #'detach' the target content from the tree used to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion will throw an error.
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x 


#A gram matrix is the result of multiplying a given matrix by its transposed matrix. 
def gram_matrix(inpt):
    a, b, c, d = inpt.size() # a = batch size = 1, b = no. feature maps, (c, d) = dimensions of a feature map (N=c*d)
    features = inpt.view(a*b, c*d)

    G = torch.mm(features, features.t()) # matrix-mult against the transposition --> gram matrix

    return G.div(a*b*c*d) # normalize the values by dividing by the number of elements in each feature map

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input 


cnn = models.vgg19(pretrained=True).features.to(device).eval() # import pretrained VGG-19 
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device) # pretrained values

# Create a module to normalize the input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1) # formatting for the expected tensor shape
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std # normalize the image 


# Configure our model
content_layers_default = ['conv_5']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img, 
                                content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0 # increment everytime we see a convolution layer
    # keep track of the anatomy of our model
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace =False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss) 
            content_losses.append(content_loss)
        
        if name in style_layers:
            # add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss) 
            style_losses.append(style_loss)

    for i in range(len(model) -1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

input_img = content_img.clone()

# add original input image to the figure
plt.figure()
imshow(input_img, title='Input Image')
#plt.savefig('Input Image')

# optimize w/ limited-memory BFGS (bunch of dudes names smashed together)
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# the transfer function
def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, 
                    input_img, num_steps=300, style_weight=100000, content_weight=0.5):
    print('Building the model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)


    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct values of updated input img
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0 

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1 
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()
            
            return style_score + content_score 
        
        optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

plt.figure()
plt.savefig('output.png')

plt.ioff()
plt.show()