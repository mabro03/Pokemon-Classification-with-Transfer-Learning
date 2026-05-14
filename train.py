import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- [1. 설정] ---
DATA_DIR = './PokemonData' 
NUM_CLASSES = 150   # 데이터셋의 실제 클래스 수에 맞춰 조절
BATCH_SIZE = 32
EPOCHS = 3          # 빠른 테스트를 위해 3으로 설정 (정확도를 높이려면 5~10 권장)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [2. 데이터 전처리 및 로드] ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(DATA_DIR):
    print(f"❌ 에러: {DATA_DIR} 경로를 찾을 수 없습니다.")
    exit()

full_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- [3. 유틸리티 함수] ---

def plot_learning_curves(history, exp_name):
    """학습 곡선을 그리고 이미지로 저장하는 함수"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss 곡선
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    plt.title(f'{exp_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy 곡선
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['test_acc'], 'ro-', label='Test Accuracy')
    plt.title(f'{exp_name} - Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{exp_name}_learning_curve.png") # 파일 저장
    # plt.show() # 화면 출력

def evaluate(model, loader):
    """모델 성능 평가 함수"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- [4. 실험 실행 함수] ---

def run_experiment(exp_config):
    print(f"\n🚀 {exp_config['name']} 실험 시작 (GPU: {torch.cuda.is_available()})")
    
    # 모델 불러오기
    weights = "DEFAULT" if exp_config["pretrained"] else None
    model = exp_config["model_func"](weights=weights)
    
    # 마지막 레이어 수정
    if "resnet" in exp_config["name"].lower():
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = evaluate(model, test_loader)
        
        history['train_loss'].append(epoch_loss)
        history['test_acc'].append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # 학습 완료 후 곡선 시각화
    plot_learning_curves(history, exp_config['name'])
    
    return model, max(history['test_acc'])

# --- [5. 메인 루프: 4가지 실험] ---

experiments = [
    {"name": "Exp1_ResNet18_Pretrained", "model_func": models.resnet18, "pretrained": True},
    {"name": "Exp2_ResNet18_Scratch", "model_func": models.resnet18, "pretrained": False},
    {"name": "Exp3_MobileNet_Pretrained", "model_func": models.mobilenet_v2, "pretrained": True},
    {"name": "Exp4_MobileNet_Scratch", "model_func": models.mobilenet_v2, "pretrained": False},
]

best_overall_acc = 0.0
results_summary = []

for exp in experiments:
    trained_model, best_acc = run_experiment(exp)
    results_summary.append((exp['name'], best_acc))
    
    # 최고 성능 모델 저장
    if best_acc >= best_overall_acc:
        best_overall_acc = best_acc
        torch.save({
            'model_name': exp['name'],
            'state_dict': trained_model.state_dict(),
            'class_names': full_dataset.classes
        }, "best_model.pth")

# --- [6. 최종 결과 요약] ---
print("\n" + "="*40)
print("📊 모든 실험 결과 요약")
print("="*40)
for name, acc in results_summary:
    print(f"{name:<30} : {acc:.2f}%")
print("="*40)
print(f"✅ 최적 모델이 'best_model.pth'로 저장되었습니다. (최고 정확도: {best_overall_acc:.2f}%)")