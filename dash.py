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

        total_trans=n['InvoiceNumber'].nunique()
        cash_trans=n[n['InvoiceType'].str.lower().str.contains('normal|online|cash')]['InvoiceNumber'].nunique()
        ins_trans=n[n['InvoiceType'].str.lower().str.contains('insurance')]['InvoiceNumber'].nunique()
        wasf_trans=n[n['InvoiceType'].str.lower().str.contains('wasfaty')]['InvoiceNumber'].nunique()
        return pd.Series({
            'total': total,
            'cash_sales': cash_sales,
            'insurance': insurance,
            'wasfaty': wasfaty,
            'total_trans':total_trans,
            'cash_trans':cash_trans,
            'ins_trans':ins_trans,
            'wasf_trans':wasf_trans
        })

    def process_data(pharmacy_data):
        data = pharmacy_data.groupby(pd.Grouper(key='InvoiceDate', freq='M')).apply(func)
        return data
    

    data = process_data(pharmacy_data)
    total_sales_col=['total','cash_sales','insurance','wasfaty']
    st.dataframe(data[total_sales_col])
    data_scaled=(data-data.min())/(data.max()-data.min())
    
    trans_col=['total_trans','cash_trans','ins_trans','wasf_trans']
    st.subheader('Visuals')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for idx, (sales, trans) in enumerate(zip(total_sales_col, trans_col)):
        ax = axes[idx]
        ax2=ax.twinx()  # محور y ثانٍ للـ transactions

    # رسم المبيعات
        sns.lineplot(data=data_scaled, x=data_scaled.index.astype(str), y=sales, ax=ax, color='blue', marker='o', label='Sales')
    # رسم عدد المعاملات
        sns.lineplot(data=data_scaled, x=data_scaled.index.astype(str), y=trans, ax=ax2, color='red', marker='x', label='Transactions')
        axes[idx].grid()
        ax.set_title(f"{sales.upper()} & {trans.upper()}", fontsize=16)
        axes[idx].tick_params(rotation=90)
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
    fig,ax =plt.subplots(figsize=(12,8))
    for col in avg_sales.columns:
        sns.lineplot(data=avg_sales,x=avg_sales.index,y=col,label=col)
    plt.grid()
    plt.legend()
    st.pyplot(fig)


    

    # زر للرجوع للصفحة الرئيسية
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))

# --- صفحة تحليل الفئات (Category Analysis) ---
elif st.session_state['page'] == 'category_analysis':
    st.markdown(f"<h1 style='text-align: center;'>📦 Category Analysis - Branch {selected_pharmacy}</h1>", unsafe_allow_html=True)
    cat_sales=pharmacy_data.groupby('InvoiceType')['ItemsNetPrice'].sum().reset_index()
    cat_sales['%']=round((cat_sales['ItemsNetPrice']/(cat_sales['ItemsNetPrice'].sum()))*100,2)
    cat_sales
    def func(n):
        ins_sales=n[n['InvoiceType'].str.lower().str.contains('insurance')]['ItemsNetPrice'].sum()
        was_sales=n[n['InvoiceType'].str.lower().str.contains('wasfaty')]['ItemsNetPrice'].sum()
        ins_trans=n[n['InvoiceType'].str.lower().str.contains('insurance')]['InvoiceNumber'].nunique()
        was_trans=n[n['InvoiceType'].str.lower().str.contains('wasfaty')]['InvoiceNumber'].nunique()
        ins_apt=ins_sales/ins_trans
        was_apt=was_sales/was_trans
        return pd.Series({'ins_sales':ins_sales,
                          'was_sales':was_sales,
                          'ins_trans':ins_trans,
                          'was_trans':was_trans,
                          'ins_apt':ins_apt,
                          'was_apt':was_apt})
    cat=pharmacy_data.groupby(pd.Grouper(key='InvoiceDate',freq='M')).apply(func)
    cat
    cat_scaled=(cat-cat.min())/(cat.max()-cat.min())
    
    fig,axes=plt.subplots(1,2,figsize=(20,6))
    axes=axes.ravel()
    sns.lineplot(data=cat_scaled,x=cat_scaled.index,y=cat_scaled['ins_sales'].rolling(window=3).mean(),ax=axes[0],label='ins_sales',marker='o')
    sns.lineplot(data=cat_scaled,x=cat_scaled.index,y=cat_scaled['ins_trans'].rolling(window=3).mean(),ax=axes[0],label='ins_trans',marker='o')
    sns.lineplot(data=cat_scaled,x=cat_scaled.index,y=cat_scaled['ins_apt'].rolling(window=3).mean(),ax=axes[0],label='ins_apt',marker='o')
    axes[0].grid()
    axes[0].legend()
    axes[0].set_title('insurance performance')

    sns.lineplot(data=cat_scaled,x=cat_scaled.index,y=cat_scaled['was_sales'].rolling(window=3).mean(),ax=axes[1],label='ins_sales',marker='o')
    sns.lineplot(data=cat_scaled,x=cat_scaled.index,y=cat_scaled['was_trans'].rolling(window=3).mean(),ax=axes[1],label='ins_trans',marker='o')
    sns.lineplot(data=cat_scaled,x=cat_scaled.index,y=cat_scaled['was_apt'].rolling(window=3).mean(),ax=axes[1],label='ins_apt',marker='o')
    axes[1].grid()
    axes[1].legend()
    axes[1].set_title('wasfaty performance')

    st.pyplot(plt)
    
    fig,ax =plt.subplots(figsize=(10,8))
    cat.corr()
    sns.heatmap(cat.corr(),annot=True)
    st.pyplot(fig)
    

    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))
    