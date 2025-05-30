import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb
import shap

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(
    page_title="再手术风险预测模型",
    page_icon="⚕️",
    layout="wide"
)

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/2222222.xlsx")
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.stop()

# 训练模型
@st.cache_data
def train_models(df):
    # 提取特征和目标变量
    X = df.drop("Unplanned reoperation", axis=1)
    y = df["Unplanned reoperation"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义模型
    models = {
        "逻辑回归 (LR)": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "随机森林 (RF)": RandomForestClassifier(random_state=42),
        "支持向量机 (SVM)": SVC(probability=True, random_state=42)
    }
    
    # 训练并评估模型
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        results[name] = {
            "准确率": accuracy,
            "F1分数": f1,
            "召回率": recall,
            "精确率": precision
        }
        
        trained_models[name] = model
    
    # 选择最佳模型 (基于F1分数)
    best_model_name = max(results, key=lambda x: results[x]["F1分数"])
    best_model = trained_models[best_model_name]
    
    return X_test, results, best_model, best_model_name, X.columns

# 主应用
def main():
    st.title("❤️ 再手术风险预测模型")
    st.markdown("本应用使用机器学习算法预测术后再手术风险，帮助医生做出更明智的决策。")
    
    # 加载数据
    df = load_data()
    
    # 显示数据概览
    with st.expander("数据概览"):
        st.write(f"数据集包含 {df.shape[0]} 条记录和 {df.shape[1]} 个特征。")
        st.dataframe(df.head())
    
    # 训练模型
    X_test, results, best_model, best_model_name, feature_names = train_models(df)
    
    # 显示模型性能
    st.subheader("模型性能比较")
    performance_df = pd.DataFrame(results).T
    st.dataframe(performance_df.style.highlight_max(axis=0))
    
    st.write(f"基于 F1 分数，**{best_model_name}** 表现最佳。")
    
    # SHAP 解释
    st.subheader("特征重要性分析")
    st.markdown("使用 SHAP (SHapley Additive exPlanations) 方法解释模型预测：")
    
    try:
        # 计算 SHAP 值
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_test)
        
        # 显示 SHAP 汇总图
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示特征重要性表格
        feature_importance = pd.Series(
            np.abs(shap_values.values).mean(axis=0), 
            index=feature_names
        ).sort_values(ascending=False)
        
        st.write("按重要性排序的特征：")
        st.dataframe(feature_importance.reset_index().rename(
            columns={"index": "特征", 0: "平均SHAP值"}
        ))
        
    except Exception as e:
        st.warning(f"SHAP 分析失败: {e}")
        st.write("可能是由于模型类型不支持SHAP自动解释。")
    
    # 用户预测界面
    st.subheader("🔍 患者再手术风险预测")
    
    with st.form("prediction_form"):
        st.markdown("### 输入患者特征")
        
        # 创建输入表单
        input_data = {}
        for feature in feature_names:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            # 根据特征类型调整输入控件
            if len(df[feature].unique()) <= 3:  # 分类特征
                input_data[feature] = st.select_slider(
                    feature,
                    options=sorted(df[feature].unique().tolist()),
                    value=mean_val if mean_val in df[feature].unique() else min_val
                )
            else:  # 连续特征
                input_data[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 20
                )
        
        # 提交预测
        submitted = st.form_submit_button("预测风险")
        
        if submitted:
            # 创建预测数据
            input_df = pd.DataFrame([input_data])
            
            # 预测
            prediction = best_model.predict(input_df)[0]
            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(input_df)[0][1]
            else:
                proba = None
            
            # 显示预测结果
            st.subheader("📊 预测结果")
            
            if prediction == 1:
                st.error(f"预测结果：**需要再手术**")
                if proba is not None:
                    st.warning(f"再手术风险概率：{proba:.2%}")
            else:
                st.success(f"预测结果：**无需再手术**")
                if proba is not None:
                    st.info(f"再手术风险概率：{proba:.2%}")
            
            # 尝试显示局部SHAP解释
            try:
                if 'explainer' in locals():
                    st.subheader("🔍 预测解释")
                    local_shap_values = explainer(input_df)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(local_shap_values[0], max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("以上图表显示了各特征对本次预测结果的贡献程度。")
            except Exception as e:
                st.warning(f"无法生成局部解释: {e}")

if __name__ == "__main__":
    main()