import torch
import torch.nn as nn
import torch.functional as F

VGG16_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG16(nn.module):
  def ___init()___(self, in_channels = 3, num_classes = 3):
    super(VGG16, self).__init__()
    
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.Conv = self.conv_layer(VGG16_architecture)
    self.FCN = nn.Sequential(nn.Linear(512*7*7, 4096),
                            nn.Relu(),
                            nn.DropOut(p=0.5),
                            
                            nn.Linear(4096, 4096),
                            nn.Relu(),
                            nn.DropOut(p=0.5),
                            
                            nn.Linear(4096, num_classes),
                            )
    
    def forward(self, x):
      x = self.Conv(x)   # Returns dim: (batch_size, 512, 7, 7)
      x = x.reshape(x.shape[0], -1) # Returns dim: (batch_size, 512*7*7)
      x = self.FCN(x) # Returns dim: (batch_size, num_classes)
    
    def conv_layer(self, architecture):
      layers = []
      in_channel = self.in_channel
      
      for x in architecture:
        if type(x) == int:
          out_channels = x
          layers += [nn.Conv2D( in_channels = in_channels, out_channels = out_channels, kernel_size = (3,3), stride= (1,1), padding = (1,1)),
                     nn.BatchNorm2d(x),
                     nn.ReLU()]
          in_channels = x
         
        elif x == 'M':
          layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
          
       return nn.Sequential(*layers)
    
    
