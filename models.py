import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


# class FemnistCNN(nn.Module):
#     """
#     Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
#     Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
#     We use `zero`-padding instead of  `same`-padding used in
#      https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
#     """
#     def __init__(self, num_classes):
#         super(FemnistCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5)

#         self.fc1 = nn.Linear(64 * 4 * 4, 2048)
#         self.output = nn.Linear(2048, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = self.output(x)
#         return x

class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
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
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
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
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


class pFedDefModel(nn.Module):
    """pFedDef model with multiple learners and attention mechanism"""
    def __init__(self, name='resnet18', n_learners=2, num_classes=10, weights=None):
        super(pFedDefModel, self).__init__()
        self.n_learners = n_learners
        self.num_classes = num_classes
        
        # Create base model
        if name == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model name: {name}")
        
        # Create learners with consistent architecture
        self.learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ) for _ in range(n_learners)
        ])
        
        # Create attention mechanisms with consistent architecture
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for _ in range(n_learners)
        ])
        
        # Initialize mixture weights
        self.mixture_weights = nn.Parameter(torch.ones(n_learners) / n_learners)
    
    def get_params(self):
        """Get model parameters for initialization"""
        return {
            'n_learners': self.n_learners,
            'num_classes': self.num_classes
        }
    
    def forward(self, x, is_training=False, client_id=None):
        """
        Forward pass
        
        Args:
            x: Input tensor
            is_training: Whether in training mode
            client_id: Specific learner ID to use (if None, use mixture)
            
        Returns:
            torch.Tensor: Model output
        """
        # Get features from base model
        features = self.base_model(x)  # [B, feature_dim]
        
        if client_id is not None:
            # Use specific learner
            output = self.learners[client_id](features)  # [B, num_classes]
            attention = torch.sigmoid(self.attention[client_id](features))  # [B, 1]
            return output * attention
        
        # Get predictions from all learners
        outputs = []
        attentions = []
        
        for i in range(self.n_learners):
            output = self.learners[i](features)  # [B, num_classes]
            attention = torch.sigmoid(self.attention[i](features))  # [B, 1]
            outputs.append(output)
            attentions.append(attention)
        
        # Stack outputs and attentions
        outputs = torch.stack(outputs, dim=0)  # [n_learners, B, num_classes]
        attentions = torch.stack(attentions, dim=0)  # [n_learners, B, 1]
        
        # Compute weighted output
        if is_training:
            # During training, use mixture weights
            weights = F.softmax(self.mixture_weights, dim=0)  # [n_learners]
            weighted_output = (outputs * weights.view(-1, 1, 1)).sum(dim=0)  # [B, num_classes]
        else:
            # During inference, use attention weights
            weights = F.softmax(attentions.squeeze(-1), dim=0)  # [n_learners, B]
            weighted_output = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, num_classes]
        
        return weighted_output
