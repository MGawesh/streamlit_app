import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
st.set_page_config(page_title="Secure Dashboard", page_icon="🔐", layout="wide")

# الباسورد من secrets
try:
    PASSWORD = st.secrets["PASSWORD"]
except Exception:
    PASSWORD="1234"

# حفظ حالة الدخول
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- واجهة تسجيل الدخول ---
if not st.session_state.authenticated:
    st.markdown("<h2 style='text-align:center;'>🔐 Enter Password to Access Dashboard</h2>", unsafe_allow_html=True)
    password = st.text_input("Password:", type="password", placeholder="Enter password here")

    if password:
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Access granted!")
            st.rerun()
        else:
            st.error("❌ Wrong password.")
    st.stop()  # يمنع ظهور أي كود للداشبورد قبل التحقق

# --- الداشبورد (يظهر فقط بعد إدخال الباسورد الصحيح) ---
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False,}))
st.markdown("<h1 style='text-align:center;'>📊 Branches Dashboard</h1>", unsafe_allow_html=True)

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
st.sidebar.title('Choose Pharmacy')

selected_pharmacy = st.sidebar.multiselect(options=pharmacy_no, label='Branch')
st.sidebar.subheader('Choose Period')
start_date=st.sidebar.date_input('Start Date',value=df['InvoiceDate'].min(),min_value=df['InvoiceDate'].min(),max_value=df['InvoiceDate'].max())
end_date=st.sidebar.date_input('End Date',value=df['InvoiceDate'].max(),min_value=df['InvoiceDate'].min(),max_value=df['InvoiceDate'].max())
if start_date > end_date :
    st.error('End Date should be after start date')
else:
    df=df[(df['InvoiceDate']>=pd.to_datetime(start_date))&(df['InvoiceDate']<=pd.to_datetime(end_date))]

if selected_pharmacy:
    pharmacy_data=df[df['BranchCode'].isin(selected_pharmacy)]
else:
    pharmacy_data=df.copy()

cash=df[df['InvoiceType'].str.lower().str.contains('normal|cash|online')]
insurance=df[df['InvoiceType'].str.lower().str.contains('insurance')]
wasfaty=df[df['InvoiceType'].str.lower().str.contains('wasfaty')]
selected_category=st.sidebar.multiselect(options=['cash','insurance','wasfaty'],label='Category')
if selected_category:
    frames=[]
    if 'cash' in selected_category:
        frames.append(cash)
    if 'insurance' in selected_category:
        frames.append(insurance)
    if 'wasfaty' in selected_category:
        frames.append(wasfaty)
    pharmacy_data=pd.concat(frames)

    
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
    daily_sales['diff']=daily_sales['ItemsNetPrice'].diff()
    daily_sales['trend']=np.sign(daily_sales['diff'])
    daily_sales['down7']=daily_sales['trend'].rolling(window=7).apply(lambda x:(x==-1).all(),raw=True)
    down7=daily_sales[daily_sales['down7']==1]

    fig = go.Figure()

# 1️⃣ خط المبيعات اليومية
    fig.add_trace(go.Scatter(
        x=daily_sales['InvoiceDate'],
        y=daily_sales['ItemsNetPrice'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='blue'),
        marker=dict(size=5)))

# 2️⃣ خطوط التحكم
    fig.add_hline(y=xbar, line=dict(color='green', dash='dash'), annotation_text="X-bar", annotation_position="top left")
    fig.add_hline(y=ucl, line=dict(color='red', dash='dash'), annotation_text="UCL", annotation_position="top left")
    fig.add_hline(y=lcl, line=dict(color='red', dash='dash'), annotation_text="LCL", annotation_position="bottom left")

# 3️⃣ النقاط الخاصة بالـ Down7
    fig.add_trace(go.Scatter(x=down7['InvoiceDate'],y=down7['ItemsNetPrice'],mode='markers',name='7 Days Down End',marker=dict(color='red', size=10, symbol='circle')))

# 4️⃣ إعدادات المخطط
    fig.update_layout(title='Daily Sales with 7 Days Down Points Highlighted',xaxis_title='Date',yaxis_title='Sales',template='plotly_white',width=1000,height=500)

# 5️⃣ عرض المخطط في Streamlit
    st.plotly_chart(fig)
   

    st.markdown('#### Days_Below_LCL')
    below_lcl=daily_sales[daily_sales['ItemsNetPrice']<=lcl]
    below_lcl
    st.markdown('#### Seven_Point_Down')
    down7

    

# نحسب المبيعات اليومية
    daily = pharmacy_data.groupby('InvoiceDate')['ItemsNetPrice'].sum().reset_index()

# Prophet لازم الأعمدة تكون بالاسمين دول تحديدًا:
    daily.rename(columns={'InvoiceDate': 'ds', 'ItemsNetPrice': 'y'}, inplace=True)

# إنشاء النموذج
    model = Prophet()

# تدريب النموذج
    model.fit(daily)

# عمل توقع لـ 30 يوم قدام مثلاً
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast
    


# رسم النتيجة
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- عرض مكونات النموذج (trend, weekly, yearly) ---
    st.subheader("🧭 Components")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)


    st.button('⬅️ Back to prevoius page',on_click=set_page,args=('statistical_process_control',))

elif st.session_state['page']=='level_three':
    st.markdown('### level 3 (SPC)')
    st.markdown('#### Root Cause Analysis')
    st.button('⬅️ Back to prevoius page',on_click=set_page,args=('statistical_process_control',))

    st.divider()  # خط فاصل بسيط
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))
