import numpy as np
import matplotlib.pyplot as plt


def kalman_filter_voltage():

    # 真实电压值（未知，假设为1.25V）
    true_voltage = 1.25
    
    # 测量噪声标准差（白噪声幅值0.1V）
    measurement_noise_std = 0.1
    
    # 过程噪声标准差（电压值恒定，过程噪声很小）
    process_noise_std = 0.00001
    
    # 卡尔曼滤波器参数
    A = 1.0  # 状态转移矩阵
    H = 1.0  # 观测矩阵
    
    # 过程噪声协方差
    Q = process_noise_std ** 2
    
    # 测量噪声协方差
    R = measurement_noise_std ** 2
    
    # 初始估计
    x_hat = 0.0  # 初始状态估计
    P = 1.0      # 初始估计协方差
    
    # 仿真参数
    n_iterations = 50  # 测量次数
    
    # 存储结果
    measurements = []
    estimates = []
    true_values = []
    prediction_errors = []
    
    # 运行卡尔曼滤波器
    for k in range(n_iterations):
        # 生成测量值（真实值 + 白噪声）
        measurement = true_voltage + np.random.normal(0, measurement_noise_std)
        measurements.append(measurement)
        true_values.append(true_voltage)
        
        # 预测步骤
        # x_hat_minus = A * x_hat
        x_hat_minus = A * x_hat
        
        # P_minus = A * P * A' + Q
        P_minus = A * P * A + Q
        
        # 更新步骤
        # K = P_minus * H' * (H * P_minus * H' + R)^(-1)
        K = P_minus * H / (H * P_minus * H + R)
        
        # x_hat = x_hat_minus + K * (y - H * x_hat_minus)
        x_hat = x_hat_minus + K * (measurement - H * x_hat_minus)
        
        # P = (1 - K * H) * P_minus
        P = (1 - K * H) * P_minus
        
        # 保存估计值
        estimates.append(x_hat)
        prediction_errors.append(np.sqrt(P))
    
    return measurements, estimates, true_values, prediction_errors


def plot_results(measurements, estimates, true_values, prediction_errors):
    """
    绘制卡尔曼滤波结果
    """
    plt.figure(figsize=(12, 8))
    
    # 子图1：测量值、估计值和真实值
    plt.subplot(2, 1, 1)
    iterations = range(len(measurements))
    
    plt.plot(iterations, measurements, 'r.', label='测量值（含噪声）', markersize=8)
    plt.plot(iterations, estimates, 'b-', label='卡尔曼滤波估计值', linewidth=2)
    plt.plot(iterations, true_values, 'g--', label='真实电压值', linewidth=2)
    
    plt.xlabel('测量次数', fontsize=12)
    plt.ylabel('电压 (V)', fontsize=12)
    plt.title('卡尔曼滤波器估计恒定电压值', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子图2：估计误差
    plt.subplot(2, 1, 2)
    estimation_errors = [abs(est - true) for est, true in zip(estimates, true_values)]
    
    plt.plot(iterations, estimation_errors, 'b-', label='估计误差', linewidth=2)
    plt.plot(iterations, prediction_errors, 'r--', label='预测误差标准差', linewidth=2)
    
    plt.xlabel('测量次数', fontsize=12)
    plt.ylabel('误差 (V)', fontsize=12)
    plt.title('卡尔曼滤波器误差分析', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印最终估计结果
    print("=" * 60)
    print("卡尔曼滤波器结果分析")
    print("=" * 60)
    print(f"真实电压值:        {true_values[0]:.6f} V")
    print(f"最终估计值:        {estimates[-1]:.6f} V")
    print(f"估计误差:          {abs(estimates[-1] - true_values[0]):.6f} V")
    print(f"平均测量值:        {np.mean(measurements):.6f} V")
    print(f"测量值标准差:      {np.std(measurements):.6f} V")
    print(f"前10次平均估计值:  {np.mean(estimates[:10]):.6f} V")
    print(f"后10次平均估计值:  {np.mean(estimates[-10:]):.6f} V")
    print("=" * 60)


if __name__ == "__main__":
    # 设置随机种子以便复现结果
    np.random.seed(42)
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
    
    print("\n正在运行卡尔曼滤波器估计恒定电压值...")
    print("模型参数：")
    print("  - 状态转移方程: x_k = x_{k-1} + w_{k-1}")
    print("  - 观测方程: y_k = x_k + v_k")
    print("  - A = 1 (状态不随时间变化)")
    print("  - H = 1 (观测值是状态的直接体现)")
    print("  - 测量噪声: 0.1V 白噪声\n")
    
    # 运行卡尔曼滤波器
    measurements, estimates, true_values, prediction_errors = kalman_filter_voltage()
    
    # 绘制结果
    plot_results(measurements, estimates, true_values, prediction_errors)
