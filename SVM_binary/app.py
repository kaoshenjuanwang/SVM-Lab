import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from svm_utils import (
    load_dataset, 
    train_svm_with_visualization, 
    plot_classification_process,
    plot_confusion_matrix, 
    evaluate_model,
    reduce_to_2d,
    visualize_pca_process,
    visualize_3d_svm
)
import time
import re

# 设置页面配置
st.set_page_config(
    page_title="SVM二分类动态演示教学平台",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/SVMmuti',
        'Report a bug': "https://github.com/yourusername/SVMmuti/issues",
        'About': "# SVM二分类动态演示教学平台\n 一个交互式的机器学习教学工具"
    }
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        background-image: url('images/ai.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        width: 1300px;
        margin: 0 auto;
        padding: 0;
        background: transparent;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px;
    }
    .header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        color: #34495e;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .progress-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .stSelectbox {
        margin-top: -0.5rem;
    }
    .stSelectbox > label {
        margin-top: -0.5rem;
    }
    div[data-baseweb="select"] {
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 设置页面标题和介绍
st.markdown('<h1 class="header">🎓 SVM二分类动态演示教学平台</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
    <p style='font-size: 1.2rem; color: #2c3e50;'>
        欢迎使用SVM二分类动态演示教学平台！本平台旨在帮助您深入理解支持向量机（SVM）在二分类问题中的应用。
        通过交互式实验和可视化，您可以：
    </p>
    <ul style='font-size: 1.1rem; color: #34495e;'>
        <li>观察不同核函数对分类边界的影响</li>
        <li>理解支持向量的选择过程</li>
        <li>直观感受分类超平面的形成过程</li>    
        <li>探索参数C和gamma及其他可能参数对模型的影响</li>
        <li>可视化决策函数值的变化</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 创建侧边栏
with st.sidebar:
    st.header("参数设置")
    
    # 数据集选择
    dataset_choice = st.selectbox(
        "选择数据集",
        ["线性可分数据集", "非线性可分数据集1（月牙形）", "非线性可分数据集2（椭圆形）", "自定义数据集"],
        help="线性可分数据集：两类数据可以用一条直线分开\n非线性可分数据集1：月牙形数据分布\n非线性可分数据集2：椭圆形数据分布\n自定义数据集：上传您自己的数据"
    )
    
    # 如果选择自定义数据集，显示文件上传器
    uploaded_file = None
    if dataset_choice == "自定义数据集":
        uploaded_file = st.file_uploader(
            "上传CSV或TXT文件（最后一列应为目标变量，且只能包含两个类别）",
            type=['csv', 'txt'],
            help="支持的文件格式：\n- CSV文件（逗号分隔）\n- TXT文件（支持逗号、制表符或空格分隔）\n\n文件要求：\n1. 所有特征必须是数值型\n2. 最后一列为目标变量（支持0/1、-1/1或1/2格式）\n3. 不能包含缺失值（或将自动删除包含缺失值的行）"
        )
        if uploaded_file is None:
            st.warning("请上传数据文件")
            st.stop()
    
    # 核函数选择
    kernel = st.selectbox(
        "核函数",
        ["linear", "rbf", "poly", "sigmoid"],
        help="选择SVM使用的核函数类型"
    )
    
    # 根据核函数类型显示相应的参数
    C = st.slider(
        "惩罚参数 C",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="控制模型对错误分类的惩罚程度，值越大分类边界越严格"
    )
    
    if kernel in ["rbf", "sigmoid"]:
        gamma = st.slider(
            f"{kernel.upper()}核参数 gamma",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help=f"控制{kernel.upper()}核函数的形状，值越大决策边界越复杂"
        )
    elif kernel == "poly":
        degree = st.number_input(
            "多项式核次数",
            min_value=2,
            max_value=5,
            value=3,
            help="多项式核函数的次数，值越大决策边界越复杂"
        )
        gamma = st.slider(
            "多项式核参数 gamma",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01
        )
    else:
        gamma = "scale"
        degree = 3

# 初始化会话状态
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training' not in st.session_state:
    st.session_state.training = False
if 'visualization_placeholder' not in st.session_state:
    st.session_state.visualization_placeholder = None
if 'X_2d' not in st.session_state:
    st.session_state.X_2d = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'X_train_2d' not in st.session_state:
    st.session_state.X_train_2d = None

# 加载数据
try:
    X, y, feature_names, target_names = load_dataset(dataset_choice, uploaded_file)
    
    # 更新DataFrame逻辑
    if 'df' not in st.session_state or 'current_dataset' not in st.session_state or st.session_state.current_dataset != dataset_choice:
        st.session_state.df = pd.DataFrame(X, columns=feature_names)
        st.session_state.df['类别'] = [target_names[label] for label in y]
        st.session_state.current_dataset = dataset_choice
    
    # 显示数据集基本信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 数据集信息")
    st.sidebar.write(f"样本数量: {X.shape[0]}")
    st.sidebar.write(f"特征数量: {X.shape[1]}")
    st.sidebar.write(f"类别数量: {len(np.unique(y))}")
    
    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["数据可视化", "模型训练", "性能分析", "升维处理"])
    
    # 数据可视化选项卡
    with tab1:
        st.write("### 数据集分布可视化")
        
        # 使用session_state中的DataFrame
        df = st.session_state.df
        
        # 添加手动添加样本点的功能
        st.write("#### 手动添加样本点")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            new_x = st.number_input(f"{feature_names[0]}", value=0.0, step=0.1)
        with col2:
            new_y = st.number_input(f"{feature_names[1]}", value=0.0, step=0.1)
        with col3:
            new_label = st.selectbox("类别", target_names, label_visibility="visible")
        
        if st.button("添加样本点"):
            # 将新样本点添加到DataFrame
            new_row = pd.DataFrame({
                feature_names[0]: [new_x],
                feature_names[1]: [new_y],
                '类别': [new_label]
            })
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            df = st.session_state.df
            
            # 更新X和y
            X = df[feature_names].values
            y = np.array([target_names.index(label) for label in df['类别']])
            
            st.success("样本点添加成功！")
            st.rerun()
        
        # 添加改变样本类别的功能
        st.write("#### 改变样本类别")
        st.write("在下方输入框中输入坐标来选择并切换最近样本点的类别")
        
        col1, col2 = st.columns(2)
        with col1:
            click_x = st.number_input(f"点击位置 {feature_names[0]}", value=0.0, step=0.1, key="click_x")
        with col2:
            click_y = st.number_input(f"点击位置 {feature_names[1]}", value=0.0, step=0.1, key="click_y")

        if st.button("切换最近点类别", key="switch_class"):
            # 计算点击位置到所有数据点的距离
            distances = np.sqrt(
                (df[feature_names[0]] - click_x) ** 2 + 
                (df[feature_names[1]] - click_y) ** 2
            )
            # 找到最近的点
            nearest_point_idx = distances.argmin()
            
            # 切换类别
            current_label = df.iloc[nearest_point_idx]['类别']
            new_label = target_names[1] if current_label == target_names[0] else target_names[0]
            st.session_state.df.iloc[nearest_point_idx, df.columns.get_loc('类别')] = new_label
            df = st.session_state.df
            
            # 更新X和y
            X = df[feature_names].values
            y = np.array([target_names.index(label) for label in df['类别']])
            
            # 显示成功消息
            st.success(f"""已将最近的样本点(距离={distances[nearest_point_idx]:.2f})的类别从 {current_label} 改为 {new_label}
                      \n样本点坐标: {feature_names[0]}={df.iloc[nearest_point_idx][feature_names[0]]:.2f}, 
                      {feature_names[1]}={df.iloc[nearest_point_idx][feature_names[1]]:.2f}""")
            
            # 使用st.rerun()来刷新页面
            st.rerun()
        
        # 添加删除样本点的功能
        st.write("#### 删除样本点")
        st.write("在下方输入框中输入坐标来选择并删除最近样本点")
        
        col1, col2 = st.columns(2)
        with col1:
            delete_x = st.number_input(f"删除位置 {feature_names[0]}", value=0.0, step=0.1, key="delete_x")
        with col2:
            delete_y = st.number_input(f"删除位置 {feature_names[1]}", value=0.0, step=0.1, key="delete_y")

        if st.button("删除最近点", key="delete_point"):
            # 计算点击位置到所有数据点的距离
            distances = np.sqrt(
                (df[feature_names[0]] - delete_x) ** 2 + 
                (df[feature_names[1]] - delete_y) ** 2
            )
            # 找到最近的点
            nearest_point_idx = distances.argmin()
            
            # 获取要删除的点的信息
            point_to_delete = df.iloc[nearest_point_idx]
            
            # 从DataFrame中删除该点
            st.session_state.df = df.drop(nearest_point_idx).reset_index(drop=True)
            df = st.session_state.df
            
            # 更新X和y
            X = df[feature_names].values
            y = np.array([target_names.index(label) for label in df['类别']])
            
            # 显示成功消息
            st.success(f"""已删除最近的样本点(距离={distances[nearest_point_idx]:.2f})
                      \n删除的样本点信息: {feature_names[0]}={point_to_delete[feature_names[0]]:.2f}, 
                      {feature_names[1]}={point_to_delete[feature_names[1]]:.2f}, 
                      类别={point_to_delete['类别']}""")
            
            # 使用st.rerun()来刷新页面
            st.rerun()
        
        # 创建散点图
        fig = px.scatter(
            df,
            x=feature_names[0],
            y=feature_names[1],
            color='类别',
            color_discrete_sequence=['#e74c3c', '#2ecc71'],  # 红色表示负例，绿色表示正例
            title='数据集分布',
            hover_data={
                feature_names[0]: ':.2f',
                feature_names[1]: ':.2f',
                '类别': True
            }
        )
        
        # 更新布局
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=600,
            xaxis=dict(
                title=dict(
                    text=feature_names[0],
                    font=dict(size=14)
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0'
            ),
            yaxis=dict(
                title=dict(
                    text=feature_names[1],
                    font=dict(size=14)
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0'
            ),
            legend=dict(
                title='类别',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示当前数据信息
        st.write("### 当前数据信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("样本数量", len(df))
        with col2:
            st.metric("特征数量", len(feature_names))
        with col3:
            st.metric("类别数量", len(df['类别'].unique()))
    
    # 模型训练选项卡
    with tab2:
        st.write("### 模型训练过程")
        st.write("点击下方按钮开始训练模型，观察分类边界的形成过程。")
        
        # 训练控制按钮
        start_button = st.button("开始训练", type="primary")
        
        # 进度条
        progress_bar = st.progress(0)
        
        # 可视化占位符
        plot_placeholder = st.empty()
        
        # 训练统计信息容器
        stats_container = st.empty()

        def update_visualization(step_data):
            """更新可视化和统计信息显示"""
            # 更新进度条
            progress_bar.progress(step_data['progress'])
            
            # 更新可视化
            with plot_placeholder:
                st.plotly_chart(step_data['fig'], use_container_width=True)
            
            # 更新训练统计信息
            with stats_container:
                st.markdown(f"""
                <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;'>
                    <h3 style='color: #2c3e50; text-align: left; margin-bottom: 20px;'>训练统计信息</h3>
                    <div style='display: flex; justify-content: space-between; gap: 20px;'>
                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                            <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>训练进度</h4>
                            <div style='font-size: 24px; font-weight: bold; color: #2980b9;'>{step_data['progress']:.0%}</div>
                        </div>
                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                            <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>当前分类准确率</h4>
                            <div style='font-size: 24px; font-weight: bold; color: #27ae60;'>{step_data['stats']['accuracy']:.2%}</div>
                        </div>
                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                            <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>支持向量数量</h4>
                            <div style='font-size: 24px; font-weight: bold; color: #8e44ad;'>{step_data['stats']['n_support_vectors']}</div>
                        </div>
                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                            <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>决策函数范围</h4>
                            <div style='font-size: 24px; font-weight: bold; color: #c0392b;'>[{step_data['stats']['decision_min']:.2f}, {step_data['stats']['decision_max']:.2f}]</div>
                        </div>
                    </div>
                    <div style='margin-top: 20px; background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                        <h4 style='color: #2c3e50; margin: 0 0 15px 0;'>当前步骤描述</h4>
                        <div style='line-height: 1.8; color: #34495e;'>
                            {format_description(step_data['stats']['description'])}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        def format_description(description):
            """格式化描述文本"""
            description_lines = description.strip().split('•')
            formatted_lines = []
            for line in description_lines:
                if line.strip():
                    line = re.sub(r'(\d+)', r'<span style="color: #e74c3c; font-weight: bold;">\1</span>', line)
                    formatted_lines.append(f"• {line.strip()}")
            return "<br>".join(formatted_lines)
        
        # 回放控制
        if 'training_history' not in st.session_state:
            st.session_state.training_history = []
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        
        # 回放控制按钮
        replay_col1, replay_col2, replay_col3 = st.columns([1, 1, 2])
        with replay_col1:
            if st.button("播放训练过程") and len(st.session_state.training_history) > 0:
                st.session_state.current_step = 0
                for step_data in st.session_state.training_history:
                    st.session_state.current_step += 1
                    update_visualization(step_data)
                    time.sleep(0.5)
        
        with replay_col2:
            if st.button("下一步") and st.session_state.current_step < len(st.session_state.training_history):
                st.session_state.current_step += 1
                update_visualization(st.session_state.training_history[st.session_state.current_step - 1])
        
        with replay_col3:
            # 确保最大值至少为1
            max_steps = max(1, len(st.session_state.training_history))
            step_slider = st.slider("训练步骤", 
                                  min_value=0,
                                  max_value=max_steps,
                                  value=min(st.session_state.current_step, max_steps))
            
            if len(st.session_state.training_history) > 0:
                if step_slider != st.session_state.current_step:
                    st.session_state.current_step = step_slider
                    if step_slider > 0:
                        update_visualization(st.session_state.training_history[step_slider - 1])
        
        if start_button:
            # 重置训练历史
            st.session_state.training_history = []
            st.session_state.current_step = 0
            
            # 定义回调函数
            def update_progress(fig, progress, stats):
                # 保存当前步骤的数据
                step_data = {
                    'fig': fig,
                    'progress': progress,
                    'stats': stats
                }
                st.session_state.training_history.append(step_data)
                st.session_state.current_step = len(st.session_state.training_history)
                
                # 更新可视化
                update_visualization(step_data)
            
            # 训练模型
            clf = train_svm_with_visualization(
                X, y,
                kernel=kernel,
                C=C,
                gamma=gamma if kernel in ["rbf", "poly"] else "scale",
                degree=degree if kernel == "poly" else 3,
                visualization_callback=update_progress
            )
            
            # 保存模型到会话状态
            st.session_state.clf = clf
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.target_names = target_names

    # 性能分析选项卡
    with tab3:
        if 'clf' in st.session_state:
            st.write("### 模型性能分析")
            
            # 划分测试集
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X,
                st.session_state.y,
                test_size=0.2,
                random_state=42
            )
            
            # 评估模型
            accuracy, y_pred = evaluate_model(
                st.session_state.clf,
                X_test,
                y_test
            )
            
            # 显示准确率
            st.metric(
                "测试集准确率",
                f"{accuracy:.2%}"
            )
            
            # 显示混淆矩阵
            fig_cm = plot_confusion_matrix(
                y_test,
                y_pred,
                st.session_state.target_names
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info('请先在"模型训练"选项卡中训练模型')

    # 升维处理选项卡
    with tab4:
        if dataset_choice in ["非线性可分数据集1（月牙形）", "非线性可分数据集2（椭圆形）", "自定义数据集"]:
            st.write("### 数据升维可视化")
            st.write("""
            在处理非线性可分的数据时，我们可以通过核函数将数据点映射到更高维的空间，使其在高维空间中变得线性可分。
            这里我们将展示数据在三维空间中的分布情况，以及SVM在高维空间中的决策边界。
            
            #### 核函数的作用
            - RBF 核函数：将数据点映射到无限维的特征空间
            - 多项式核函数：将数据映射到更高次的特征空间
            - Sigmoid核函数：模拟神经网络的激活函数特性
            
            下面的3D可视化展示了：
            1. 原始数据点在升维后空间中的分布（红色和绿色点）
            2. 支持向量（黑色圆圈）
            3. 决策超平面（彩色曲面）
            """)
            
            # 添加核函数选择和参数调整
            col1, col2, col3 = st.columns(3)
            with col1:
                kernel = st.selectbox(
                    "选择核函数",
                    ["rbf", "poly", "sigmoid"],
                    help="选择不同的核函数来观察数据在高维空间的映射效果"
                )
            
            with col2:
                C = st.slider(
                    "惩罚参数 C",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="控制模型对错误分类的容忍度，值越大表示越不容忍错误"
                )
            
            with col3:
                if kernel in ["rbf", "poly", "sigmoid"]:
                    gamma = st.slider(
                        "核函数参数 gamma",
                        min_value=0.01,
                        max_value=10.0,
                        value=1.0,
                        step=0.01,
                        help="控制核函数的形状，值越大决策边界越复杂"
                    )
                else:
                    gamma = "scale"
            
            # 创建3D可视化
            fig_3d, clf_3d = visualize_3d_svm(X, y, kernel=kernel, C=C, gamma=gamma)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 显示模型信息
            st.write("#### 模型信息")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("支持向量数量", len(clf_3d.support_vectors_))
            with col2:
                y_pred = clf_3d.predict(X)
                accuracy = np.mean(y_pred == y)
                st.metric("训练集准确率", f"{accuracy:.2%}")
            with col3:
                st.metric("特征维度", f"2 → ∞" if kernel == "rbf" else f"2 → 3")
            
            st.write("""
            #### 升维原理解释
            在这个三维可视化中，我们可以观察到：
            1. 原本在二维平面中无法线性分开的数据点，在三维空间中变得可分。
            2. 决策超平面（彩色曲面）清晰地将两类数据分开。
            3. 支持向量（黑色圆圈）位于决策边界附近，是确定最优分类面的关键点。
            
            这种可视化帮助我们理解核技巧的本质：通过将数据映射到更高维的空间，
            使得原本在低维空间中非线性可分的数据变得线性可分。虽然实际的 RBF 核
            会将数据映射到无限维空间，但通过这个三维的可视化，我们可以直观地理
            解这个过程。
            """)
        else:
            st.info('升维处理功能仅在使用"非线性可分数据集"或"自定义数据集"时可用。请在左侧选择相应的数据集来体验此功能。')

except Exception as e:
    st.error(f"发生错误: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>© 2025  SVM 二分类动态演示教学平台 | 由大连理工大学 MindForge 团队开发</p>
    <p style='font-size: 0.8rem;'>版本 2.1.6</p>
</div>
""", unsafe_allow_html=True)