import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pharmacy Dashboard", layout="wide")
st.title('Main KPIs')

# --- تحميل البيانات ---
@st.cache_data
def load_data():
    df = pd.read_excel(r'total.xlsx',parse_dates=['InvoiceDate'])
    return df

df = load_data()

# --- اختيار الفرع ---
pharmacy_no = df['BranchCode'].unique().tolist()
st.sidebar.title('Branch')
selected_pharmacy = st.sidebar.selectbox(options=pharmacy_no, label='Choose pharmacy')
pharmacy_data = df[df['BranchCode'] == selected_pharmacy]
st.sidebar.markdown('[collect data](https://script.google.com/macros/s/AKfycbzXwJ4ExBn6bjPc7RUDmDqlTDfU9Q9dHO1BnjkCICAFAJsaidL8br7RPZZZSDtKP6hf/exec)')
# --- تهيئة session_state ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# --- دوال لتبديل الصفحات ---
def set_page(page_name):
    st.session_state['page'] = page_name

# --- القائمة الرئيسية (Home Page) ---
if st.session_state['page'] == 'home':
    st.markdown(f"<h1 style='text-align: center;'>Branch Number {selected_pharmacy}</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.button("🕒 Time Series Analysis", on_click=set_page, args=('time_series',))
    with col2:
        st.button("📦 Category Analysis", on_click=set_page, args=('category_analysis',))

# --- صفحة تحليل السلاسل الزمنية ---
elif st.session_state['page'] == 'time_series':
    st.markdown(f"<h1 style='text-align: center;'>📈 Time Series Analysis - Branch {selected_pharmacy}</h1>", unsafe_allow_html=True)
    st.markdown("### Sales Category")

    # -------- التحليل الأول (Sales Category) --------
    def func(n):
        total = n['ItemsNetPrice'].sum()
        cash = n[n['InvoiceType'].str.lower().str.contains('normal|online|cash')]
        cash_sales = cash['ItemsNetPrice'].sum()
        insurance = n[n['InvoiceType'].str.lower().str.contains('insurance')]['ItemsNetPrice'].sum()
        wasfaty = n[n['InvoiceType'].str.lower().str.contains('wasfaty')]['ItemsNetPrice'].sum()
        return pd.Series({
            'total': total,
            'cash_sales': cash_sales,
            'insurance': insurance,
            'wasfaty': wasfaty
        })

    def process_data(pharmacy_data):
        data = pharmacy_data.groupby(pd.Grouper(key='InvoiceDate', freq='M')).apply(func)
        return data

    data = process_data(pharmacy_data)
    st.dataframe(data)

    st.subheader('Visuals')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for idx, col in enumerate(data.columns):
        sns.lineplot(data=data, x=data.index.astype(str), y=col, ax=axes[idx])
        axes[idx].grid()
        axes[idx].set_title(col.upper(), fontsize=18)
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown('----')
    st.subheader('SALES KPIs')

    # -------- التحليل الثاني (KPIs) --------
    def func_kpi(n):
        cash = n[n['InvoiceType'].str.lower().str.contains('normal|cash|online')]
        apt = cash['ItemsNetPrice'].sum() / cash['InvoiceNumber'].nunique()
        rsp = cash['ItemsNetPrice'].sum() / cash['Quantity'].sum()
        bsvolume = cash['Quantity'].sum() / cash['InvoiceNumber'].nunique()
        transno = cash['InvoiceNumber'].nunique()
        return pd.Series({
            'APT': apt.round(2),
            'RSP': rsp.round(2),
            'B.S.VOLUME': bsvolume.round(2),
            'TRANS. NO.': transno
        })

    def process_kpi(pharmacy_data):
        kpis = pharmacy_data.groupby(pd.Grouper(key='InvoiceDate', freq='M')).apply(func_kpi)
        return kpis

    kpis = process_kpi(pharmacy_data)
    st.dataframe(kpis)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for idx, col in enumerate(kpis.columns):
        sns.lineplot(data=kpis, x=kpis.index, y=col, ax=axes[idx])
        axes[idx].grid()
        axes[idx].set_title(col, fontsize=18)
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown('----')
    st.subheader('PHARMACIST PERFORMANCE')
    st.text('Avg_daily_sales_for_every_pharmacist')

    def func(n):
        avg_daily=n['ItemsNetPrice'].sum()/n['InvoiceDate'].nunique()
        return pd.Series({'avg_daily':avg_daily})
    avg_sales=pharmacy_data.groupby([pd.Grouper(key='InvoiceDate',freq='M'),'SalesName']).apply(func).unstack()
    avg_sales
    

# رسم خط لكل صيدلي
    fig,ax =plt.subplots(figsize=12,8)
    for col in avg_sales.columns:
        sns.lineplot(data=avg_sales,x=avg_sales.index,y=col)
    
    st.pyplot(fig)


    

    # زر للرجوع للصفحة الرئيسية
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))

# --- صفحة تحليل الفئات (Category Analysis) ---
elif st.session_state['page'] == 'category_analysis':
    st.markdown(f"<h1 style='text-align: center;'>📦 Category Analysis - Branch {selected_pharmacy}</h1>", unsafe_allow_html=True)
    st.info("🚧 سيتم إضافة تحليل الفئات لاحقًا...")
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))
