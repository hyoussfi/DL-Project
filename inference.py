import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the CNN model (from Base implementation)
class SkinCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkinCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define MobileNetV3 model
def create_mobilenetv3_model(num_classes):
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

# Define EfficientNet model
def create_efficientnet_model(num_classes):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 768),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(768, 384),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(384, num_classes)
    )
    return model

class SkinDiseasePredictor:
    def __init__(self, model_path, model_type='cnn', num_classes=10):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Updated class names matching the Kaggle dataset structure
        self.class_names = [
            'Eczema',
            'Melanoma',
            'Atopic Dermatitis',
            'Basal Cell Carcinoma (BCC)',
            'Melanocytic Nevi (NV)',
            'Benign Keratosis-like Lesions (BKL)',
            'Psoriasis pictures, Lichen Planus and related diseases',
            'Seborrheic Keratoses and other Benign Tumors',
            'Tinea Ringworm Candidiasis and other Fungal Infections',
            'Warts Molluscum and other Viral Infections'
        ]
        
        # Initialize model
        if model_type == 'cnn':
            self.model = SkinCNN(num_classes)
        elif model_type == 'mobilenetv3':
            self.model = create_mobilenetv3_model(num_classes)
        elif model_type == 'efficientnet':
            self.model = create_efficientnet_model(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load and prepare the model
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        top3_prob, top3_indices = torch.topk(probabilities, 3, dim=1)
        top3_predictions = [
            {
                'class': self.class_names[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top3_prob[0], top3_indices[0])
        ]
            
        return {
            'predicted_class': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'top3_predictions': top3_predictions,
            'all_probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }

def main():
    image_path = "path_to_your_image.jpg"  # Replace with your image
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    
    models = {
        'CNN': (os.path.join(base_dir, "Base", "skin_disease_cnn.pth"), 'cnn'),
        'MobileNetV3': (os.path.join(base_dir, "MobileNetv3", "best_skin_disease_mobilenetv3.pth"), 'mobilenetv3'),
        'EfficientNet': (os.path.join(base_dir, "EfficientNet", "best_skin_disease_efficientnetb0.pth"), 'efficientnet')
    }
    
    for model_name, (model_path, model_type) in models.items():
        try:
            print(f"\nLoading {model_name} from: {model_path}")
            predictor = SkinDiseasePredictor(model_path, model_type)
            prediction = predictor.predict(image_path)
            
            print(f"\n{model_name} Predictions:")
            for i, pred in enumerate(prediction['top3_predictions'], 1):
                print(f"{i}. {pred['class']}: {pred['confidence']:.4f}")
        except Exception as e:
            print(f"\nError loading {model_name}: {str(e)}")

if __name__ == "__main__":
    main()
