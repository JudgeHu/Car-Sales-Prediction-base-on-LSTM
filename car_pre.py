import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Computing with the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置调用GPU，如果无法调用GPU，则使用CPU

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Retrieve Data
data = pd.read_excel('dataset.xlsx')

# Normalization process
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['sales'].values.reshape(-1, 1))


# Define functions to perform data conversion
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


# Creating supervised learning data for training and test sets
look_back = 12
train_X, train_Y = create_dataset(scaled_data, look_back)
# Converting data to tensor
train_X = torch.from_numpy(train_X).type(torch.Tensor).to(device)
train_Y = torch.from_numpy(train_Y).type(torch.Tensor).to(device)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = out.transpose(1, 2)
        out = self.bn(out)
        out = out.transpose(1, 2)
        out = self.dropout1(out)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)  # 修改这里
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)  # 修改这里

        out, (hn, cn) = self.lstm1(out, (h0.detach(), c0.detach()))
        out = self.dropout2(out)
        out = self.fc2(out[:, -1, :])
        out = out.unsqueeze(1).repeat(1, x.size(1), 1)
        out, (hn, cn) = self.lstm2(out, (h0.detach(), c0.detach()))
        out = self.dropout3(out)
        out = self.fc3(out[:, -1, :])
        return out


# Defining model parameters
input_size = 1
hidden_size = 120
num_layers = 3
output_size = 1
learning_rate = 0.001
num_epochs = 1500

# Defining the model, loss function and optimiser
lstm = LSTM(input_size, hidden_size, num_layers, output_size)
lstm = lstm.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Training Model
train_loss = []
for epoch in range(num_epochs):
    outputs = lstm(train_X)
    optimizer.zero_grad()
    loss = criterion(outputs, train_Y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    train_loss.append(loss.item())

# Inverse normalization
train_predict = lstm(train_X)
train_predict = scaler.inverse_transform(train_predict.cpu().detach().numpy())
train_predict = np.round(train_predict)
train_Y = scaler.inverse_transform(train_Y.cpu().detach().numpy())


# Calculating Mean Absolute Percentage Error
def mape(y_true, y_pred):
    mape = []
    for i in range(len(y_true)):
        mape_ = np.abs((y_true[i] - y_pred[i]) / y_true[i])
        mape.append(mape_)
    return np.array(mape), np.mean(mape)


train_real = np.array(data['sales'][look_back:])
train_predict = np.array(train_predict)
mape_, mean_mape = mape(train_real, train_predict)
print('Mean Absolute Percentage Error:', mean_mape)

# Plot model loss diagram
plt.plot(train_loss, color='darkorange')
plt.xlabel('Number of Trainings')
plt.ylabel('Training Losses')
plt.title('LSTM Model Loss Function Plot')
plt.show()

# Output prediction results to excel
output = pd.DataFrame(
    {'时间': data['time'].iloc[look_back:], '每期绝对误差': mape_.reshape(-1), '每期预测值': train_predict.reshape(-1)})
output.to_excel('模型预测.xlsx', index=False)

# Use the last 12 periods of data as initial input
test_input = scaled_data[-look_back:]
predictions = []
# Forecasting future sales and updating the inputs for the next forecasting period
for i in range(20):
    # Convert data to tensor and add two new dimensions（batch 和 feature）
    test_input = torch.from_numpy(test_input).type(torch.Tensor).to(device)
    test_input = test_input.unsqueeze(0)  # Add dimension 1 to the first and last dimension
    # Model prediction
    lstm.eval()  # Setting up the model for evaluation mode
    with torch.no_grad():
        prediction = lstm(test_input)
    prediction = prediction.cpu().numpy()[0][0]  # Convert to a NumPy array and take the first value.
    predictions.append(prediction)
    # Update the inputs for the next forecasting period
    test_input = np.concatenate([test_input.cpu().squeeze().detach().numpy()[1:], [prediction]]).reshape(-1, 1)
# Remove the added dimension and add the predicted value at the end

# Inverse normalization to get the final predictions
predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)
predictions = predictions.astype(int)

# Mapping of projected results
plt.plot(pd.date_range(start='2016-01', end='2023-05', freq='M'), train_real, label='Actual Value')
plt.plot(pd.date_range(start='2016-01', end='2023-05', freq='M'), train_predict, label='Predicted Value in Past')
plt.plot(pd.date_range(start='2023-04', end='2024-12', freq='M'), predictions, label='Predicted Value in Future')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.title('Comparison of Actual and Predicted Values of LSTM Model')
plt.legend()
plt.show()
print('Sales in the Next 20 Month：\n', predictions)
