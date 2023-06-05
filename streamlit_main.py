import streamlit as st
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair
from sklearn.decomposition import PCA
from SNNDPC import SNNDPC
from numpy import min, max
from scipy.stats import gaussian_kde


@st.cache_data
def load1():
    data_1 = pd.read_csv("CreditCard.csv")
    data_1.dropna(subset=["CREDIT_LIMIT"], inplace=True)
    data_1.drop(["CUST_ID"], axis=1, inplace=True)
    data_1["MINIMUM_PAYMENTS"].fillna(data_1["MINIMUM_PAYMENTS"].median(), inplace=True)
    return data_1

def show1():

    data = load1()
    
    column_names = data.columns.tolist()
    lis = [data[[s]] for s in column_names]

    st.subheader(" ")
    st.subheader("总体分布直方图")
    tabs = (tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17,) = st.tabs(
        [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N","O","P","Q",]
    )

    i = 0

    for tab in tabs:
        with tab:
            #tab.subheader("总体分布直方图")

            da = lis[i]
            chart = (
                altair.Chart(da)
                .mark_bar()
                .encode(
                    x=altair.X(
                        column_names[i], bin=altair.Bin(maxbins=20), title=column_names[i]
                    ),
                    y="count()",
                )
            )
            st.altair_chart(chart.interactive(), use_container_width=True)
        i += 1

@st.cache_data
def load2():
    DA=pd.read_csv('CreditCard.csv')
    DA.dropna(subset=['CREDIT_LIMIT'], inplace=True)
    DA.drop(['CUST_ID'], axis=1, inplace=True)
    DA.drop(['PURCHASES'],axis=1,inplace=True)
    DA.drop(['PURCHASES_INSTALLMENTS_FREQUENCY'], axis=1, inplace=True)
    DA.drop(['CASH_ADVANCE_FREQUENCY'], axis=1, inplace=True)
    DA['MINIMUM_PAYMENTS'].fillna(DA['MINIMUM_PAYMENTS'].median(), inplace=True)

    dat=DA.values[5000:6000]
    return dat
    
def show2():
    dat = load2()
    
    dat = PCA(n_components=6).fit_transform(dat)
    dat = (dat - min(dat, axis=0)) / (max(dat, axis=0) - min(dat, axis=0))



    _, _, centers, labels = SNNDPC(7, 3, dat)


    df = pd.DataFrame(dat,columns=["x","y","z","a",'b','c'])
    center_df = df.loc[centers]
    single_df = df.loc[[35]]
    
    

    #st.write(center_df)

    df['label'] = labels

    center_df['label'] = [0,1,2]
    center_df['font'] = [3,3,3]

    single_df['label'] = ["Europe"]
    single_df['font'] = [3]




    chart = altair.Chart(df).mark_point(shape='diamond').encode(
    x='c',
    y='z',
    color = 'label'
    ).interactive()

    center_chart = altair.Chart(center_df).mark_point().encode(
    x='c',
    y='z',
    color = "label",
    size = 'font'
    ).interactive()

    sig_chart = altair.Chart(single_df).mark_point(shape='cross').encode(
    x='c',
    y='z',
    color = "label",
    size = 'font'
    ).interactive()

    st.subheader("")
    st.subheader("用户聚类")
    st.altair_chart(chart+center_chart+sig_chart,theme="streamlit",use_container_width=True)


def M():
    data = pd.read_csv("CreditCard.csv")
    data.dropna(subset=["CREDIT_LIMIT"], inplace=True)
    data.drop(["CUST_ID"], axis=1, inplace=True)
    data["MINIMUM_PAYMENTS"].fillna(data["MINIMUM_PAYMENTS"].median(), inplace=True)

    column_names = data.columns.tolist()
    lis = [data[[s]] for s in column_names]

    tag1 = 0
    tag2 = 0

    st.header("用户信息分布统计")
    with st.sidebar:
        global csv
        uploaded_file = st.file_uploader("请选择客户信息文件", help="excel")
        if uploaded_file is not None:
            csv_data = pd.read_csv(uploaded_file)
            csv = csv_data
            #st.write(csv.iloc[0,1])
            # st.write(dataframe.loc[0, :])
            # st.write(csv_data)
            st.header(csv_data.iloc[0, 0])

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "余额",
                csv_data.iloc[0, 1],"-" if data['PAYMENTS'].mean()-csv_data.iloc[0, 14]>0 else "+"
            )
            col1.metric("总消费", csv_data.iloc[0, 14], "-" if data['BALANCE'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col1.metric("可用预付现金", csv_data.iloc[0, 6],"-" if data['CASH_ADVANCE'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col1.metric("分期采购金额", csv_data.iloc[0, 5],"-" if data['INSTALLMENTS_PURCHASES'].mean()-csv_data.iloc[0, 14]>0 else "+")

            col2.metric("余额更新率", csv_data.iloc[0, 2], "-" if data['BALANCE_FREQUENCY'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col2.metric("购买频率", csv_data.iloc[0, 7], "-" if data['PURCHASES_FREQUENCY'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col2.metric("预付现金频率", csv_data.iloc[0, 10], "-" if data['CASH_ADVANCE_FREQUENCY'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col2.metric("分期购买频率", csv_data.iloc[0, 9], "-" if data['PURCHASES_INSTALLMENTS_FREQUENCY'].mean()-csv_data.iloc[0, 14]>0 else "+")

            col3.metric("信用卡额度", csv_data.iloc[0, 13], "-" if data['CREDIT_LIMIT'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col3.metric("预付现金交易次数", csv_data.iloc[0, 11], "-" if data['CASH_ADVANCE_TRX'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col3.metric("全额付款比率", str(csv_data.iloc[0, 16]) + "%", "-" if data['PRC_FULL_PAYMENT'].mean()-csv_data.iloc[0, 14]>0 else "+")
            col3.metric("一次购买最高金额", csv_data.iloc[0, 4], "-" if data['ONEOFF_PURCHASES'].mean()-csv_data.iloc[0, 14]>0 else "+")
            # mianinfo()
            st.subheader(" ")
            analys_button = st.button("分析", use_container_width=True)
            if analys_button:
                tag1 = 1

            classfisy_button = st.button("聚类",use_container_width=True)
            if classfisy_button:
                tag1 = 1
                tag2 = 1

    data_1 = load1()

    column_names_1 = data_1.columns.tolist()

    st.header(" ")
    st.subheader("用户所处位置信息")
    tabs = (tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17,) = st.tabs(
    [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N","O","P","Q",]
)
    i = 0

    for tab in tabs:
        with tab:
            #tab.subheader("用户所处位置")
            
            fig,ax = plt.subplots()
            data = np.random.randn(1000)  # 随机生成一组数据

# 使用高斯核估计密度函数
            kde = gaussian_kde(data)

# 生成一组横轴数据
            x = np.linspace(data.min(), data.max(), 100)

# 计算对应的核密度值
            density = kde(x)

# 绘制核密度曲线
            ax.plot(x, density, label='Density')

# 填充核密度曲线下方区域
            ax.fill_between(x, density, color='skyblue', alpha=0.5)
    
            st.pyplot(fig)
        i += 1

    if tag1 == 1:
        show1()
    if tag2 == 1:
        show2()



M()







    # col1, col2 = st.columns(2)
    # with col1:
    #     st.button("常规按钮")
    #     st.text_input("文本输入组件样式：", "默认输入内容")
    #     st.selectbox("单选择组件样式：", ("默认选项1", "默认选项2", "默认选项3"))
    # with col2:
    #     st.download_button("下载按钮", "欢迎使用应用创建工具")
    #     st.number_input("数字输入组件样式：", 1, 20, 10)
    #     st.multiselect("多选框组件样式：", ["默认选项1", "默认选项2", "默认选项3"], ["默认选项1"])
    # st.markdown("<hr />", unsafe_allow_html=True)
    # st.write("**2、媒体组件**")
    # st.write("Streamlit支持多种媒体组件，包括图片、视频、语音等，以下为常用的输入组件样式：")

# st.markdown("<hr />", unsafe_allow_html=True)
# st.write("**3、风格迁移交互示例教程**")
# st.write("利用PaddleHub中AnimeGAN模型将输入图片转换成新海诚动漫风格的交互效果，可以通过以下方式展现：")
# per_image = st.file_uploader("上传图片", type=["png", "jpg"], label_visibility="hidden")
# col3, col4 = st.columns(2)
# with col3:
#     if per_image:
#         st.image(per_image)
#     else:
#         st.image("https://codelab-public.bj.bcebos.com/base.jpeg")
#     test = st.button("提交图片")
# with col4:
#     if test:
#         st.image("https://codelab-public.bj.bcebos.com/test.jpeg")
#         test = False
#     else:
#         st.write("暂无预测结果,请点击提交图片")
# st.markdown("<hr />", unsafe_allow_html=True)
# st.write("**以上是推荐常用交互样式哦，还在等什么，赶快行动起来，结合飞桨模型打造你的专属应用吧！**")
