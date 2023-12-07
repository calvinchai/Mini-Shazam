import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import librosa
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        
        return torch.sigmoid(self.linear(x))

def load_model(n_features):
    model = LogisticRegression(n_features)
    model.load_state_dict(torch.load('model/saved_model.pth'))
    model.eval().to(device)
    return model

def save_model(model):
    torch.save(model.state_dict(), 'model/saved_model.pth')

def train_model(X_train, y_train):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = LogisticRegression(X_train.size(1)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(model.parameters(), lr=1)
    
    def closure():
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    save_model(model)
    return model

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.round(outputs)
        print(preds)
        print(y_test)
        accuracy = y_test.eq(preds).sum() / float(len(y_test))
        # accuracy = accuracy_score(y_test.cpu(), preds.cpu())
    return accuracy

model = None
def predict(input_array):
    n_features = input_array.shape[1]

    global model
    if model is None:
        model = load_model(n_features)
    input_tensor = torch.tensor(input_array, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(input_tensor)
    result = torch.round(output)

    return result

def predict_audio(file):
    f,sr = librosa.load(file)
    feature = librosa.feature.melspectrogram(y=f)
    # mean 
    feature = np.mean(feature, axis=1)
    input = torch.tensor(feature, dtype=torch.float32, device=device).expand(1, -1)
    result = predict(input)
    result = int(result.cpu()[0][0])
    return result

    

def main():
    data = np.load("model/data.npz")
    X_train = torch.tensor(data["Xtr"], dtype=torch.float, device=device)  
    y_train = torch.tensor(data["Ytr"], dtype=torch.float, device=device)  
    X_test = torch.tensor(data["Xte"], dtype=torch.float, device=device)  
    y_test = torch.tensor(data["Yte"], dtype=torch.float, device=device)  
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    
    model = train_model(X_train, y_train)
    train_accuracy = evaluate_model(model, X_train, y_train)
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
