import torch
import torch.nn as nn

class DCMAC(nn.Module):
    def __init__(self, gen_factor, num_weights, input_min, input_max, sample_input):
        super(DCMAC, self).__init__()
        self.gen_factor = gen_factor
        self.num_weights = num_weights
        self.input_min = input_min
        self.input_max = input_max
        self.weight_vec = nn.Parameter(torch.ones(num_weights))
        # 处理关联映射
        self.process_association_map(sample_input, input_min, input_max)

    def get_assoc_vec_ind(self, input_val, num_of_assoc_vec, input_min, input_max):
        proportion_ind = (num_of_assoc_vec-2) * ((input_val - input_min)/(input_max - input_min)) + 1
        proportion_ind = torch.where(proportion_ind < 1, 1, proportion_ind)
        proportion_ind = torch.where(proportion_ind > num_of_assoc_vec-1, num_of_assoc_vec-1, proportion_ind)
        return proportion_ind
    
    def process_association_map(self, input_data, input_min, input_max):
        # 计算关联向量的数量
        num_of_assoc_vec = self.num_weights + 1 - self.gen_factor
        # 获取数据值对应的关联向量索引
        assoc_vec_ind = self.get_assoc_vec_ind(input_data, num_of_assoc_vec, input_min, input_max)
        # 将数据值和对应的索引存入关联映射中
        self.association_map_keys=input_data
        self.association_map_values=torch.floor(assoc_vec_ind).int()

    def forward(self, input_data):
        # 计算关联向量的数量
        num_of_assoc_vec = self.num_weights + 1 - self.gen_factor
        # 获取数据值对应的关联向量索引
        asc_wt_ind = torch.floor(self.get_assoc_vec_ind(input_data, num_of_assoc_vec, self.input_min, self.input_max)).int()
        # 创建一个用于批量计算的索引矩阵
        indices = asc_wt_ind.unsqueeze(1) + torch.arange(self.gen_factor).unsqueeze(0).to(input_data.device)
        # 计算输出值
        y_output = torch.sum(self.weight_vec[indices], dim=1)
        return y_output
        
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # 定义2π的常量
    TWO_PI = 2 * np.pi
    # 生成从0到2π的100个等间距样本点
    sample_input = np.linspace(0, TWO_PI, 1000)
    # 计算每个样本点的正弦值
    sample_output = np.sin(sample_input)

    # 将输入和输出数据合并成一个二维数组
    total_data = np.column_stack((sample_input, sample_output))
    # 随机打乱数据
    np.random.shuffle(total_data)

    # 取前70个数据作为训练数据
    train_data = total_data[:700]
    # 取后30个数据作为测试数据
    test_data = total_data[700:]

    x_train = torch.tensor(train_data[:, 0], dtype=torch.float32).cuda()
    y_train = torch.tensor(train_data[:, 1], dtype=torch.float32).cuda()
    x_test = torch.tensor(test_data[:, 0], dtype=torch.float32).cuda()
    y_test = torch.tensor(test_data[:, 1], dtype=torch.float32).cuda()

    num_weights = 35  # 定义权重的数量，即关联向量的长度
    gen_factor = 10  # 定义泛化因子，即每个输入激活的关联向量数目

    # 创建DCMAC对象
    dcmac = DCMAC(gen_factor, num_weights, 0, TWO_PI, x_train).cuda()

    # 定义优化器
    optimizer = torch.optim.Adam(dcmac.parameters(), lr=0.001)
    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 训练模型
    for epoch in range(100000):
        optimizer.zero_grad()
        y_pred = dcmac(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 测试模型
    with torch.no_grad():
        y_pred_test = dcmac(x_test)
        test_loss = loss_fn(y_pred_test, y_test)
        print(f'Test Loss: {test_loss.item()}')

    # 绘制结果
    # 对测试结果进行排序
    sorted_indices = x_test.argsort()
    x_test_sorted = x_test[sorted_indices]
    y_pred_test_sorted = y_pred_test[sorted_indices]

    plt.plot(sample_input, sample_output, label="Original Curve")
    plt.plot(x_test_sorted.cpu(), y_pred_test_sorted.cpu(), color='red', label="DCMAC Prediction")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.savefig("torch_dcmac.png")

