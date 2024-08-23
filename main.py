import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class GPAPredictor(nn.Module):
    def __init__(self):
        super(GPAPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
def load_data(file_path):
    columns_gpa=['GPA1', 'GPA2'] #edit the number of GPA of each sem (i had only two so I used two)
    df = pd.read_csv(file_path)
    X = df[columns_gpa].values
    y = (df['GPA2'] + np.random.normal(0, 0.1, df.shape[0])).clip(0, 10).values.reshape(-1, 1) 
    names = df['Name'].values
    return X, y, names

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    model = GPAPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 5000
    best_val_loss = float('inf')
    patience = 200
    counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    model.load_state_dict(torch.load('best_model.pth'))
    return model, scaler_X, scaler_y

def predict_gpa(model, scaler_X, scaler_y, name, X, names):
    if name in names:
        index = list(names).index(name)
        student_data = X[index]
        student_data_scaled = scaler_X.transform([student_data])
        input_tensor = torch.FloatTensor(student_data_scaled)
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
        prediction = scaler_y.inverse_transform(prediction_scaled.numpy())
        return prediction[0][0]
    else:
        return None
    
def main():
    file_path = 'data.csv'
    X, y, names = load_data(file_path)
    model, scaler_X, scaler_y = train_model(X, y)
    while True:
        name = input("Enter student name (or 'q' to quit): ")
        if name.lower() == 'q':
            break
        predicted_gpa = predict_gpa(model, scaler_X, scaler_y, name, X, names)
        if predicted_gpa is not None:
            print(f"Predicted next GPA for {name}: {predicted_gpa:.2f}")
        else:
            print(f"Student {name} not found in the database.")

if __name__ == "__main__":
    main()