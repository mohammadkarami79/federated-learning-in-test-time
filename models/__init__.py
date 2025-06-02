"""
Models package
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .pfeddef_model import pFedDefModel as LocalpFedDefModel

# Define models directly to avoid circular imports
class FemnistCNN(nn.Module):
    """CNN for FEMNIST dataset"""
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 800)
        self.output = nn.Linear(800, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 dataset"""
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class NextCharacterLSTM(nn.Module):
    """LSTM for next character prediction"""
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)
        return output

class LinearLayer(nn.Module):
    """Simple linear layer"""
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)

# Factory functions for pre-trained models
def get_vgg11(n_classes):
    """Create VGG11 model with n_classes outputs"""
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    return model

def get_squeezenet(n_classes):
    """Create SqueezeNet model with n_classes outputs"""
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes
    return model

def get_mobilenet(n_classes):
    """Create MobileNet model with n_classes outputs"""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model

def get_resnet18(n_classes):
    """Create ResNet18 model with n_classes outputs"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

def get_resnet34(n_classes):
    """Create ResNet34 model with n_classes outputs"""
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

class pFedDefModel(nn.Module):
    """pFedDef model with multiple learners and attention mechanism"""
    def __init__(self, name='resnet18', n_learners=2, num_classes=10, weights=None):
        super(pFedDefModel, self).__init__()
        self.n_learners = n_learners
        self.num_classes = num_classes
        
        if name == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model name: {name}")
        
        self.learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ) for _ in range(n_learners)
        ])
        
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for _ in range(n_learners)
        ])
        
        self.mixture_weights = nn.Parameter(torch.ones(n_learners) / n_learners)
    
    def get_params(self):
        return {'n_learners': self.n_learners, 'num_classes': self.num_classes}
    
    def forward(self, x, is_training=False, client_id=None):
        features = self.base_model(x)
        
        if client_id is not None:
            output = self.learners[client_id](features)
            attention = torch.sigmoid(self.attention[client_id](features))
            return output * attention
        
        outputs = []
        attentions = []
        
        for i in range(self.n_learners):
            output = self.learners[i](features)
            attention = torch.sigmoid(self.attention[i](features))
            outputs.append(output)
            attentions.append(attention)
        
        outputs = torch.stack(outputs, dim=0)
        attentions = torch.stack(attentions, dim=0)
        
        if is_training:
            weights = F.softmax(self.mixture_weights, dim=0)
            weighted_output = (outputs * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            weights = F.softmax(attentions.squeeze(-1), dim=0)
            weighted_output = (outputs * weights.unsqueeze(-1)).sum(dim=0)
        
        return weighted_output

def get_model(cfg):
    """
    Get model based on configuration
    
    Args:
        cfg: Configuration object with model settings
        
    Returns:
        torch.nn.Module: Initialized model
    """
    n_classes = getattr(cfg, 'N_CLASSES', 10)
    model_type = getattr(cfg, 'MODEL_TYPE', 'resnet18').lower()
    width = getattr(cfg, 'RESNET_WIDTH', 1.0)
    
    if model_type == 'cifar10cnn':
        return CIFAR10CNN(n_classes)
    elif model_type == 'femnistcnn':
        return FemnistCNN(n_classes)
    elif model_type == 'vgg11':
        return get_vgg11(n_classes)
    elif model_type == 'squeezenet':
        return get_squeezenet(n_classes)
    elif model_type == 'mobilenet':
        return get_mobilenet(n_classes)
    elif model_type == 'resnet18':
        model = get_resnet18(n_classes)
        if width != 1.0:
            model = scale_model_width(model, width)
        return model
    elif model_type == 'resnet34':
        model = get_resnet34(n_classes)
        if width != 1.0:
            model = scale_model_width(model, width)
        return model
    elif model_type == 'pfeddef':
        n_learners = getattr(cfg, 'FEDEM_N_LEARNERS', 3)
        return pFedDefModel(n_learners=n_learners, num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def scale_model_width(model, width_factor):
    """Scale model width by the given factor - DISABLED for stability"""
    # Disable model scaling to avoid channel mismatch issues
    # This requires a complete rewrite to scale ALL layers consistently
    if width_factor != 1.0:
        import warnings
        warnings.warn(f"Model width scaling disabled to avoid channel mismatch. Using width=1.0 instead of {width_factor}")
    
    # Return original model unchanged
    return model

def get_model_params(model):
    """Get number of parameters in model"""
    return sum(p.numel() for p in model.parameters())

def get_trainable_params(model):
    """Get number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

__all__ = [
    'pFedDefModel', 'LocalpFedDefModel', 'get_model', 
    'get_model_params', 'get_trainable_params',
    'FemnistCNN', 'CIFAR10CNN', 'NextCharacterLSTM', 'LinearLayer',
    'get_vgg11', 'get_squeezenet', 'get_mobilenet', 'get_resnet18', 'get_resnet34'
] 