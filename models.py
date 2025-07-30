import torch
import torch.nn as nn
import torchvision.models as models
from attention_modules import SpatialAttention, ChannelAttention, CBAMBlock




class ResNet18(nn.Module):
    def __init__(self, in_channels=16, num_classes=9, dropout_p=0.0):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # changing the input layer to recive the number of input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self._add_dropout_to_blocks(dropout_p)

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def _add_dropout_to_blocks(self, p):
        for name, module in self.base_model.named_children():
            if 'layer' in name:
                for block in module:
                    block.dropout = nn.Dropout(p=p)



    def forward(self, x):
        return self.base_model(x)
    



class ResNet34(nn.Module):
    def __init__(self, in_channels=16, num_classes=9, dropout_p=0.0):
        super().__init__()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # changing the input layer to recive the number of input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self._add_dropout_to_blocks(dropout_p)

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def _add_dropout_to_blocks(self, p):
        for name, module in self.base_model.named_children():
            if 'layer' in name:
                for block in module:
                    block.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.base_model(x)
    



class ResNet18WithSpatialAttention(nn.Module):
    def __init__(self, in_channels=16, num_classes=9, dropout_p=0.0):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # changing the input layer to recive the number of input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self._add_dropout_to_blocks(dropout_p)

        # extracting the features before the attention 
        self.features = nn.Sequential(*list(self.base_model.children())[:-2])

        # adding spatial attention
        self.attention = SpatialAttention()

        # average pool --> fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(512)  
        self.fc = nn.Linear(512, num_classes)

    def _add_dropout_to_blocks(self, p):
        for name, module in self.base_model.named_children():
            if 'layer' in name:
                for block in module:
                    block.dropout = nn.Dropout(p=p)
                    original_forward = block.forward

                    def forward_with_dropout(x, block=block, orig_fwd=original_forward):
                        out = orig_fwd(x)
                        return block.dropout(out)

                    block.forward = forward_with_dropout

    def forward(self, x):
        x = self.features(x)
        attn = self.attention(x)
        x = x * attn
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)    
    



class ResNet34WithSpatialAttention(nn.Module):
    def __init__(self, in_channels=16, num_classes=9, dropout_p=0.0):
        super().__init__()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # changing the input layer to recive the number of input channels
        self.base_model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self._add_dropout_to_blocks(dropout_p)

        # extracting the features before the attention
        self.features = nn.Sequential(*list(self.base_model.children())[:-2])

        # adding spatial attention
        self.attention = SpatialAttention()

        # average pool --> fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(512)  
        self.fc = nn.Linear(512, num_classes)

    def _add_dropout_to_blocks(self, p):
        for name, module in self.base_model.named_children():
            if 'layer' in name:
                for block in module:
                    block.dropout = nn.Dropout(p=p)
                    original_forward = block.forward

                    def forward_with_dropout(x, block=block, orig_fwd=original_forward):
                        out = orig_fwd(x)
                        return block.dropout(out)

                    block.forward = forward_with_dropout

    def forward(self, x):
        x = self.features(x)
        attn = self.attention(x)
        x = x * attn
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)              
        return self.fc(x)
    



class ResNet18WithRepeatedSpatialAttention(nn.Module):
    def __init__(self, in_channels=16, num_classes=9, dropout_p=0.0):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # changing the input layer to recive the number of input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # dividing the layers to add attention modules to each residual block
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.dropout_p = dropout_p
        self._add_dropout_to_blocks()

        # adding spatial attention
        self.att1 = SpatialAttention()
        self.att2 = SpatialAttention()
        self.att3 = SpatialAttention()
        self.att4 = SpatialAttention()

        # average pool --> fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)

    def _add_dropout_to_blocks(self):
        if self.dropout_p > 0:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    block.dropout = nn.Dropout(p=self.dropout_p)
                    original_forward = block.forward

                    def forward_with_dropout(x, block=block, orig_fwd=original_forward):
                        out = orig_fwd(x)
                        return block.dropout(out)

                    block.forward = forward_with_dropout

    def forward(self, x):
        x = self.conv1(x)       
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     

        x = self.layer1(x)      
        x = x * self.att1(x)

        x = self.layer2(x)      
        x = x * self.att2(x)

        x = self.layer3(x)      
        x = x * self.att3(x)

        x = self.layer4(x)      
        x = x * self.att4(x)

        x = self.avgpool(x)     
        x = torch.flatten(x, 1) 
        return self.fc(x)
    



class ResNet18WithCBAM(nn.Module):
    def __init__(self, in_channels=16, num_classes=9, dropout_p=0.0):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # changing the input layer to recive the number of input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = base.bn1
        self.relu  = base.relu
        self.maxpool = base.maxpool

        # adding CBAM modules in each residual block 
        self.layer1 = base.layer1
        self.cbam1 = CBAMBlock(64)

        self.layer2 = base.layer2
        self.cbam2 = CBAMBlock(128)

        self.layer3 = base.layer3
        self.cbam3 = CBAMBlock(256)

        self.layer4 = base.layer4
        self.cbam4 = CBAMBlock(512)

        # average pool --> fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_out = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)

        self._add_dropout_to_blocks(dropout_p)

    def _add_dropout_to_blocks(self, p):
        for module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in module:
                block.dropout = nn.Dropout(p=p)
                original_forward = block.forward

                def forward_with_dropout(x, block=block, orig_fwd=original_forward):
                    out = orig_fwd(x)
                    return block.dropout(out)

                block.forward = forward_with_dropout

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn_out(x)
        return self.fc(x)