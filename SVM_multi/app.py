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
    visualize_pca_process
)

# 设置页面配置
st.set_page_config(
    page_title="SVM完成多分类任务教学平台",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/SVMmuti',
        'Report a bug': "https://github.com/yourusername/SVMmuti/issues",
        'About': "# SVM完成多分类任务教学平台\n 一个交互式的机器学习教学工具"
    }
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        width: 1300px;
        margin: 0 auto;
        padding: 0;
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
</style>
""", unsafe_allow_html=True)

# 设置页面标题和介绍
st.markdown('<h1 class="header">🎓 SVM完成多分类任务教学平台</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
    <p style='font-size: 1.2rem; color: #2c3e50;'>
        欢迎使用SVM完成多分类任务教学平台！本平台旨在帮助您深入理解支持向量机（SVM）在多分类问题中的应用。
        通过交互式实验和可视化，您可以：
    </p>
    <ul style='font-size: 1.1rem; color: #34495e;'>
        <li>探索不同核函数的效果</li>
        <li>调整SVM参数并观察结果</li>
        <li>直观感受SVM将未知样本分类的过程</li>
        <li>多分类场景下分类超平面及支持向量的动态确定过程</li>    
        <li>可视化决策边界</li>
        <li>分析模型性能</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 创建侧边栏
with st.sidebar:
    st.header("参数设置")
    
    # 数据集选择
    dataset_choice = st.selectbox(
        "选择数据集",
        ["鸢尾花数据集", "Glass数据集", "Wine数据集", "自定义数据集"]
    )
    
    # 如果选择自定义数据集，显示文件上传器
    uploaded_file = None
    if dataset_choice == "自定义数据集":
        uploaded_file = st.file_uploader(
            "上传CSV文件（最后一列应为目标变量）",
            type=['csv']
        )
        if uploaded_file is None:
            st.warning("请上传数据文件")
            st.stop()
    
    # 核函数选择
    kernel = st.selectbox(
        "核函数",
        ["rbf", "linear", "poly", "sigmoid"],
        help="选择SVM使用的核函数类型"
    )
    
    # 根据核函数类型显示相应的参数
    C = st.slider(
        "惩罚参数 C",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="控制模型对错误分类的惩罚程度"
    )
    
    if kernel in ["rbf", "sigmoid"]:
        gamma = st.slider(
            f"{kernel.upper()}核参数 gamma",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help=f"控制{kernel.upper()}核函数的形状"
        )
    elif kernel == "poly":
        degree = st.number_input(
            "多项式核次数",
            min_value=2,
            max_value=5,
            value=3,
            help="多项式核函数的次数"
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
    
    # 显示数据集基本信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 数据集信息")
    st.sidebar.write(f"样本数量: {X.shape[0]}")
    st.sidebar.write(f"特征数量: {X.shape[1]}")
    st.sidebar.write(f"类别数量: {len(np.unique(y))}")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["数据可视化", "模型训练", "性能分析"])
    
    # 数据可视化选项卡
    with tab1:
        st.write("### 数据预处理")
        # 使用PCA进行降维
        pca_figures, X_pca, components = visualize_pca_process(X, feature_names)
        
        # 显示每个独立的图表
        for fig in pca_figures:
            st.plotly_chart(fig, use_container_width=True)
        
        # 显示降维后的数据
        st.write("#### 降维后的数据预览")
        # 创建一个格式化的DataFrame
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        # 设置索引从1开始
        df_pca.index = df_pca.index + 1
        # 格式化数值，保留4位小数
        df_pca = df_pca.round(4)
        # 使用styler来美化显示
        styled_df = df_pca.style.set_properties(**{
            'background-color': '#f5f5f5',
            'color': '#2c3e50',
            'border': '1px solid #ddd',
            'padding': '8px',
            'text-align': 'right'
        }).set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#3498db'),
                      ('color', 'white'),
                      ('font-weight', 'bold'),
                      ('text-align', 'center'),
                      ('padding', '8px')]},
            {'selector': 'tr:nth-of-type(odd)',
             'props': [('background-color', '#ffffff')]},
            {'selector': 'tr:hover',
             'props': [('background-color', '#eaf2f8')]}
        ]).hide(axis='index')
        
        # 显示带样式的DataFrame
        st.dataframe(styled_df, height=300)
    
    # 模型训练选项卡
    with tab2:
        st.write("### 模型训练过程")
        st.write("点击下方按钮开始训练模型，观察分类边界的形成过程。")
        
        # 训练按钮
        if st.button("开始训练", type="primary"):
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            plot_placeholder = st.empty()
            stats_container = st.empty()  # 新增：用于显示训练统计信息的容器
            
            # 定义回调函数
            def update_progress(result, progress):
                fig, info = result
                progress_bar.progress(progress)
                status_text.write(f"训练进度: {progress:.0%}")
                with plot_placeholder:
                    st.plotly_chart(fig, use_container_width=True)
                
                # 解析迭代信息
                info_lines = info.split('\n')
                step_data = {
                    'progress': progress,
                    'stats': {
                        'accuracy': float(info_lines[1].split('：')[1].strip('%')) / 100,
                        'best_accuracy': float(info_lines[2].split('：')[1].strip('%')) / 100,
                        'n_support_vectors': int(info_lines[4].split('：')[1].split('个')[0]),
                        'description': info
                    }
                }
                
                # 更新训练统计信息
                with stats_container:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;'>
                        <h3 style='color: #2c3e50; text-align: left; margin-bottom: 20px;'>训练统计信息</h3>
                        <div style='display: flex; flex-direction: column; gap: 20px;'>
                            <div style='display: flex; justify-content: space-between; gap: 20px;'>
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                                    <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>训练进度</h4>
                                    <div style='font-size: 24px; font-weight: bold; color: #2980b9;'>{step_data['progress']:.0%}</div>
                                </div>
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                                    <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>当前训练准确率</h4>
                                    <div style='font-size: 24px; font-weight: bold; color: #27ae60;'>{step_data['stats']['accuracy']:.2%}</div>
                                </div>
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                                    <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>最佳准确率</h4>
                                    <div style='font-size: 24px; font-weight: bold; color: #c0392b;'>{step_data['stats']['best_accuracy']:.2%}</div>
                                </div>
                            </div>
                            <div style='display: flex; justify-content: space-between; gap: 20px;'>
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                                    <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>已标记样本</h4>
                                    <div style='font-size: 24px; font-weight: bold; color: #8e44ad;'>{int(info_lines[0].split('：')[1].split('个')[0])}</div>
                                </div>
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                                    <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>错误分类点数</h4>
                                    <div style='font-size: 24px; font-weight: bold; color: #fb8f83;'>{int(info_lines[3].split('：')[1].split('个')[0])}</div>
                                </div>
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; flex: 1;'>
                                    <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>支持向量数量</h4>
                                    <div style='font-size: 24px; font-weight: bold; color: #ffc500;'>{step_data['stats']['n_support_vectors']}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
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

except Exception as e:
    st.error(f"发生错误: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>© 2025  SVM 完成多分类任务教学平台 | 由大连理工大学 MindForge 团队开发</p>
    <p style='font-size: 0.8rem;'>版本 2.2.3</p>
</div>
""", unsafe_allow_html=True) 