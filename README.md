# SVM Virtual Lab

这是一个基于SVM（支持向量机）的虚拟仿真实验平台，用于AI辅助教学。平台提供了交互式的SVM学习环境，让学生能够直观地理解SVM的工作原理。

## 功能特点

- 交互式数据生成
- 实时SVM模型训练
- 可视化决策边界
- 支持向量可视化
- 实时预测演示

## 技术栈

- 前端：React + Plotly.js
- 后端：Python Flask
- 机器学习：scikit-learn
- 数据可视化：Plotly.js

## 安装说明

### 后端设置

1. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行后端服务器：
```bash
python app.py
```

### 前端设置

1. 进入前端目录：
```bash
cd frontend
```

2. 安装依赖：
```bash
npm install
```

3. 运行开发服务器：
```bash
npm start
```

## 使用说明

1. 点击"Generate New Data"按钮生成新的随机数据点
2. 点击"Train Model"按钮训练SVM模型
3. 观察决策边界的变化
4. 可以多次生成新数据并重新训练，观察不同数据分布下的决策边界

## 教学应用

本平台可用于以下教学场景：

1. SVM基本原理教学
2. 决策边界可视化
3. 支持向量的理解
4. 核函数的作用演示
5. 分类问题的实践练习

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License 