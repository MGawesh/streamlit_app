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
    df = pd.read_excel('total.xlsx',parse_dates=['InvoiceDate'])
    return df

df = load_data()

# --- اختيار الفرع ---
pharmacy_no = df['BranchCode'].unique().tolist()
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
st.sidebar.title('Branch')
selected_pharmacy = st.sidebar.multiselect(options=pharmacy_no, label='Choose pharmacy')
if selected_pharmacy:
    pharmacy_data=df[df['BranchCode'].isin(selected_pharmacy)]
else:
    pharmacy_data=df.copy()

st.sidebar.markdown('[Collect Data](https://script.google.com/macros/s/AKfycbzXwJ4ExBn6bjPc7RUDmDqlTDfU9Q9dHO1BnjkCICAFAJsaidL8br7RPZZZSDtKP6hf/exec)')
st.sidebar.markdown('[Shortage Tracking](https://script.google.com/macros/s/AKfycby-GiNZC5T3-WoIuhgD-Dxbl9xKOg_wm2cRChhfrim5TRqWYyRnLhwILxginVwIvzgSkw/exec)')
st.sidebar.markdown('[Contact Branches](https://mgawesh.github.io/contact_us/)')
# --- تهيئة session_state ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# --- دوال لتبديل الصفحات ---
def set_page(page_name):
    st.session_state['page'] = page_name

# --- القائمة الرئيسية (Home Page) ---
if st.session_state['page'] == 'home':
    st.markdown(f"<h1 style='text-align: center;'>Branch Number {selected_pharmacy}</h1>", unsafe_allow_html=True)
    col1, col2,col3 = st.columns(3)
    with col1:
        st.button("🕒 Time Series Analysis", on_click=set_page, args=('time_series',))
    with col2:
        st.button("📦 Category Analysis", on_click=set_page, args=('category_analysis',))
    with col3:
        st.button("📈 statistical process control", on_click=set_page, args=('statistical_process_control',))

 

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

    def func(n):
        total = n['ItemsNetPrice'].sum()
        cash = n[n['InvoiceType'].str.lower().str.contains('normal|online|cash')]
        cash_sales = ((cash['ItemsNetPrice'].sum())/total)*100
        insurance = ((n[n['InvoiceType'].str.lower().str.contains('insurance')]['ItemsNetPrice'].sum())/total)*100
        wasfaty = ((n[n['InvoiceType'].str.lower().str.contains('wasfaty')]['ItemsNetPrice'].sum())/total)*100

        total_trans=n['InvoiceNumber'].nunique()
        cash_trans=n[n['InvoiceType'].str.lower().str.contains('normal|online|cash')]['InvoiceNumber'].nunique()
        ins_trans=n[n['InvoiceType'].str.lower().str.contains('insurance')]['InvoiceNumber'].nunique()
        wasf_trans=n[n['InvoiceType'].str.lower().str.contains('wasfaty')]['InvoiceNumber'].nunique()
        return pd.Series({
            'total': total,
            'cash_sales%': cash_sales,
            'insurance%': insurance,
            'wasfaty%': wasfaty,
            'total_trans%':total_trans,
            'cash_trans%':cash_trans,
            'ins_trans%':ins_trans,
            'wasf_trans%':wasf_trans
        })

    def process_data(pharmacy_data):
        dataper = pharmacy_data.groupby(pd.Grouper(key='InvoiceDate', freq='M')).apply(func)
        return dataper
    

    dataper = process_data(pharmacy_data)
    dataper[['total','cash_sales%','insurance%','wasfaty%']]
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
    

elif st.session_state['page'] == 'statistical_process_control':

    st.markdown("<h2 style='text-align:center;'>📊 Statistical Process Control</h2>", unsafe_allow_html=True)
    st.markdown("#### level :")

    # تقسيم الصفحة إلى 3 أعمدة
    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("🧩 Level 1", use_container_width=True, on_click=set_page, args=('level_one',))
    with col2:
        st.button("📈 Level 2", use_container_width=True, on_click=set_page, args=('level_two',))
    with col3:
        st.button("📊 Level 3", use_container_width=True, on_click=set_page, args=('level_three',))
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))
elif st.session_state['page']=='level_one':
    st.markdown('Level One (SPC)')
    def func(n):
        n=n.sort_values('InvoiceDate')
        last_date=n['InvoiceDate'].max()
        last_date_revenu=n[n['InvoiceDate']==last_date]['ItemsNetPrice'].sum()
        day_of_month=last_date.day
        last_month=last_date.month
        last_year=last_date.year
        current_mtd=n[(n['InvoiceDate'].dt.year==last_year)&
                      (n['InvoiceDate'].dt.month==last_month)&
                      (n['InvoiceDate'].dt.day<=day_of_month)]
        revenu_current=current_mtd['ItemsNetPrice'].sum()

        if last_month==1:
            prev_month=12
            prev_year=last_year-1
        else:
            prev_month=last_month-1
            prev_year=last_year
        prev_mtd=n[(n['InvoiceDate'].dt.month==prev_month)&
                   (n['InvoiceDate'].dt.year==prev_year)&
                   (n['InvoiceDate'].dt.day<=day_of_month)]
        revenu_prev=prev_mtd['ItemsNetPrice'].sum()
        if revenu_prev != 0:
            mom=((revenu_current-revenu_prev)/revenu_prev)*100
        else:
            mom=None
        seven_day_MA=n.groupby(pd.Grouper(key='InvoiceDate',freq='D'))['ItemsNetPrice'].sum().rolling(window=7).mean().iloc[-1]
        deviation=((last_date_revenu-seven_day_MA)/last_date_revenu)*100
        
        return pd.Series({'last_day_revenu':last_date_revenu,'revenue_MTD':revenu_current,'MOM%':mom,
                          '7day_moving_average':seven_day_MA,
                          'Deviation':deviation})
    stat=df.groupby('BranchCode').apply(func)
    stat
    st.button("⬅️ Back to previous page", on_click=set_page, args=('statistical_process_control',))
elif st.session_state['page']=='level_two':
    st.markdown('### level 2 (SPC)')
    st.markdown('#### I-MR (Individual Moving Range)')
    daily_sales=pharmacy_data.groupby(pd.Grouper(key='InvoiceDate',freq='D'))['ItemsNetPrice'].sum().reset_index()
    daily_sales['MR']=daily_sales['ItemsNetPrice'].diff().abs()
    xbar=daily_sales['ItemsNetPrice'].mean()
    mrbar=daily_sales['MR'].mean()
    ucl=xbar+2.66*mrbar
    lcl=xbar-2.66*mrbar
    figure,axes=plt.subplots(figsize=(20,5))
    sns.lineplot(data=daily_sales,x='InvoiceDate',y='ItemsNetPrice',ax=axes)
    plt.axhline(xbar, color='green', linestyle='--', label='X-bar')
    plt.axhline(ucl, color='red', linestyle='--', label='UCL')
    plt.axhline(lcl, color='red', linestyle='--', label='LCL')
    axes.grid()
    axes.legend()
    st.pyplot(figure)
    st.markdown('#### Days_Below_LCL')
    below_lcl=daily_sales[daily_sales['ItemsNetPrice']<=lcl]
    below_lcl
    st.button('⬅️ Back to prevoius page',on_click=set_page,args=('statistical_process_control',))

elif st.session_state['page']=='level_three':
    st.markdown('### level 3 (SPC)')
    st.button('⬅️ Back to prevoius page',on_click=set_page,args=('statistical_process_control',))

    st.divider()  # خط فاصل بسيط
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))
