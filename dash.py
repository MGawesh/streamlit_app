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
def func(n):
    if len(str(n))==2:
        return f'P0{n}'
    else:
        return f'P{n}'
df['BranchCode']=df['BranchCode'].apply(func)

@st.cache_data
def load_fraction():
    fraction=pd.read_excel('fraction.xlsx')
    return fraction
fraction=load_fraction()
# --- اختيار الفرع ---
# --- تحويل التاريخ وتنظيم البيانات الأساسية ---
# --- تحويل التاريخ وتنظيم البيانات الأساسية ---
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
pharmacy_no = df['BranchCode'].unique().tolist()

# --- عناصر الفلترة في الـ Sidebar ---
st.sidebar.title('Filters')

selected_pharmacy = st.sidebar.multiselect('Choose Pharmacy', options=pharmacy_no)
start_date = st.sidebar.date_input('Start Date', value=df['InvoiceDate'].min())
end_date = st.sidebar.date_input('End Date', value=df['InvoiceDate'].max())

selected_category = st.sidebar.multiselect(
    'Select Category',
    options=['cash', 'insurance', 'wasfaty'],
    default=[]
)

# --- تحقق من التواريخ ---
if start_date > end_date:
    st.error('⚠️ End Date must be after Start Date')
else:
    # فلترة أولية بناءً على التاريخ
    filtered = df[
        (df['InvoiceDate'] >= pd.to_datetime(start_date)) &
        (df['InvoiceDate'] <= pd.to_datetime(end_date))
    ]

    # فلترة حسب نوع الفاتورة
    if selected_category:
        mask = (
            (df['InvoiceType'].str.lower().str.contains('normal|cash|online') if 'cash' in selected_category else False) |
            (df['InvoiceType'].str.lower().str.contains('insurance') if 'insurance' in selected_category else False) |
            (df['InvoiceType'].str.lower().str.contains('wasfaty') if 'wasfaty' in selected_category else False)
        )
        filtered = filtered[mask]

    # فلترة حسب الفروع
    if selected_pharmacy:
        filtered = filtered[filtered['BranchCode'].isin(selected_pharmacy)]

    # النتيجة النهائية
    pharmacy_data = filtered.copy()

    




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
    col1, col2,col3,col4 = st.columns(4)
    with col1:
        st.button("🕒 Time Series Analysis", on_click=set_page, args=('time_series',))
    with col2:
        st.button("📦 Category Analysis", on_click=set_page, args=('category_analysis',))
    with col3:
        st.button("📈 statistical process control", on_click=set_page, args=('statistical_process_control',))
    with col4:
        st.button('📦 Inventory Optimization',on_click=set_page,args=('Inventory Optimization',))

 

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

    st.markdown('-----')
    st.button('pharmacist level',on_click=set_page,args=('pharmacist_level',))
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))
elif st.session_state['page']=='pharmacist_level':
        
    st.subheader('PHARMACIST PERFORMANCE')
    st.text('Avg_daily_sales_for_every_pharmacist')

    def func(n):
        avg_daily=n['ItemsNetPrice'].sum()/n['InvoiceDate'].nunique()
        return pd.Series({'avg_daily':avg_daily})
    avg_sales=pharmacy_data.groupby([pd.Grouper(key='InvoiceDate',freq='M'),'SalesName']).apply(func).unstack()
    avg_sales
    fig,ax =plt.subplots(figsize=(10,4))
    for col in avg_sales.columns:
        sns.lineplot(data=avg_sales,x=avg_sales.index,y=col,label=col)
    plt.grid()
    plt.legend()
    st.pyplot(fig)

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
        kpis_pharmacist = pharmacy_data.groupby([pd.Grouper(key='InvoiceDate', freq='M'),'SalesName']).apply(func_kpi).unstack()
        return kpis_pharmacist

    kpis_pharmacist = process_kpi(pharmacy_data)
    st.dataframe(kpis_pharmacist)
    apt=kpis_pharmacist['APT']

    
    


# إنشاء الشكل
    fig = go.Figure()

# رسم الخطوط لكل عمود (زي ما كنت بتعمل بـ Seaborn)
    for col in apt.columns:
        fig.add_trace(go.Scatter(
            x=apt.index,
            y=apt[col],
            mode='lines+markers',
            name=col
    ))

# إضافة الخط الأفقي عند y = 87
    fig.add_hline(
        y=87,
        line_dash="dash",
        line_color="red",
        annotation_text="Target = 87",
        annotation_position="top left"
)

# تخصيص شكل الرسم
    fig.update_layout(
    title="APT Trend by Pharmacist",
    xaxis_title="Month",
    yaxis_title="APT",
    template="plotly_white",
    height=400,
    legend=dict(title="Pharmacists")
)

# عرض الرسم في Streamlit
    st.plotly_chart(fig, use_container_width=True)

    bsvolume=kpis_pharmacist['B.S.VOLUME']
    
    fig = go.Figure()
    for col in bsvolume.columns:
        fig.add_trace(go.Scatter(x=bsvolume.index,y=bsvolume[col],mode='lines+markers',name=col))
    fig.add_hline(y=2.57,line_dash='dash',line_color='red',annotation_text='target = 2.57',annotation_position='top left')
    fig.update_layout(title='B.S.VOLUME BY PHARMACIST',
                          xaxis_title='MONTH',
                          yaxis_title='B.S.VOLUME',
                          template='plotly_white',
                          height=400,
                          legend=dict(title='pharmacists'))
    st.plotly_chart(fig,use_container_width=True)

    rsp=kpis_pharmacist['RSP']
    fig=go.Figure()
    for col in rsp.columns:
        fig.add_trace(go.Scatter(x=rsp.index,y=rsp[col],mode='lines+markers',name=col))
    fig.add_hline(y=36,line_dash='dash',line_color='red',annotation_text='target = 36',annotation_position='top left')
    fig.update_layout(title='RSP',xaxis_title='month',yaxis_title='RSP',template='plotly_white',height=400,legend=dict(title=('pharmacists')))
    st.plotly_chart(fig,use_container_width=True)


    
    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('time_series',))

    
    

# رسم خط لكل صيدلي
    

    # زر للرجوع للصفحة الرئيسية
    #st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))

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

    
    st.markdown('#### PROPHET FORECAST')
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


elif st.session_state['page']=='Inventory Optimization':
    st.markdown('#### PARETO ANALYSIS')
    pareto=pharmacy_data.groupby('Name')['Quantity'].sum().reset_index().sort_values(by='Quantity',ascending=False)
    pareto['%']=(pareto['Quantity']/pareto['Quantity'].sum())*100
    pareto['cumsum']=pareto['%'].cumsum()
    pareto_items=pareto[pareto['cumsum']<=80]
    pareto_items_count=pareto_items['Name'].count()
    pareto_count=pareto['Name'].count()
    contrib=round((pareto_items_count/pareto_count)*100,2)
    st.markdown(f"#### {contrib} % of items represent 80% of sold items {[pareto_items_count]}")
    pareto_items
    

    
    @st.cache_data
    def load_stock():
        stock=pd.read_excel('209 stock.xlsx')
        return stock
    stock=load_stock()
    stock['Branch']=stock['Branch'].astype(str)
    stock=stock[['MaterialName','Branch','UnRestrictedStock']]
    
    st.markdown('#### forecast')
# تحويل رقم الفاتورة لنص (احتياطي)
    pharmacy_data['InvoiceNumber'] = pharmacy_data['InvoiceNumber'].astype(str)



# ---- تعديل fraction بكفاءة ----
    fraction_sum = fraction.groupby(['Name', 'UnitOfMeasurement'])['unit'].sum().reset_index()
# نعمل merge بدل ما نلف بـ for loop
    pharmacy_data = pharmacy_data.merge(
    fraction_sum,
    on=['Name', 'UnitOfMeasurement'],
    how='left'
)
# تحديث الكمية فقط للصفوف اللي عندها قيمة fraction
    pharmacy_data['Quantity'] = np.where(
        pharmacy_data['unit'].notna(),
        pharmacy_data['unit'],
        pharmacy_data['Quantity']
)
    pharmacy_data.drop(columns=['unit'], inplace=True)

# ---- حساب الفرق الزمني بالشهور ----
    max_date = pharmacy_data['InvoiceDate'].max()
    pharmacy_data['month'] = ((max_date - pharmacy_data['InvoiceDate']).dt.days // 30)

# ---- Pivot Table ----
    cleaned_pivot = pharmacy_data.pivot_table(
    index=['BranchCode', 'Name'],
    columns='month',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# ---- حساب المتوسطات ----
# بدلاً من 3 apply، نستخدم mean vectorized
    for i, col_range in enumerate([[0,1,2], [3,4,5], [6,7,8]], start=1):
        cols = [c for c in col_range if c in cleaned_pivot.columns]
        cleaned_pivot[f'avgq{i}'] = cleaned_pivot[cols].mean(axis=1)

# ---- Forecast بشكل أسرع ----
    avgq1, avgq2, avgq3 = cleaned_pivot['avgq1'], cleaned_pivot['avgq2'], cleaned_pivot['avgq3']


    conditions = [
    (cleaned_pivot['avgq1'] > 3 * (cleaned_pivot['avgq2'] + cleaned_pivot['avgq3']) / 2),
    (cleaned_pivot['avgq2'] > 3 * (cleaned_pivot['avgq1'] + cleaned_pivot['avgq3']) / 2),
    (cleaned_pivot['avgq3'] > 3 * (cleaned_pivot['avgq2'] + cleaned_pivot['avgq1']) / 2),
    ((cleaned_pivot['avgq1'] + cleaned_pivot['avgq2']) / 2 > cleaned_pivot['avgq3'] * 3),
    ((cleaned_pivot['avgq1'] + cleaned_pivot['avgq3']) / 2 > cleaned_pivot['avgq2'] * 3),
    ((cleaned_pivot['avgq3'] + cleaned_pivot['avgq2']) / 2 > cleaned_pivot['avgq1'] * 3)
]

    choices = [
    cleaned_pivot['avgq1'],
    cleaned_pivot['avgq2'],
    cleaned_pivot['avgq3'],
    (cleaned_pivot['avgq1'] + cleaned_pivot['avgq2']) / 2,
    (cleaned_pivot['avgq1'] + cleaned_pivot['avgq3']) / 2,
    (cleaned_pivot['avgq3'] + cleaned_pivot['avgq2']) / 2
]

# القيمة الافتراضية لو مفيش شرط اتحقق
    default = (cleaned_pivot['avgq1'] * 0.6) + (cleaned_pivot['avgq2'] * 0.25) + (cleaned_pivot['avgq3'] * 0.15)

    cleaned_pivot['forecast'] = np.select(conditions, choices, default=default)

    cleaned_pivot['dailyforecast'] = cleaned_pivot['forecast'] / 30


    conditions = [
    cleaned_pivot['forecast'] > 50,
    cleaned_pivot['forecast'] >= 6,
    cleaned_pivot['forecast'] < 6
]
    min_values = [cleaned_pivot['dailyforecast'] * 16,
              cleaned_pivot['dailyforecast'] * 20,
              cleaned_pivot['dailyforecast'] * 35]
    max_values = [cleaned_pivot['dailyforecast'] * 24,
              cleaned_pivot['dailyforecast'] * 28,
              cleaned_pivot['dailyforecast'] * 50]

    cleaned_pivot['min'] = np.select(conditions, min_values)
    cleaned_pivot['max'] = np.select(conditions, max_values)

# النتيجة النهائية
    cleaned_pivot = cleaned_pivot.reset_index()
    cleaned_pivot['BranchCode']=cleaned_pivot['BranchCode'].astype(str)
    cleaned_pivot[['BranchCode','Name','forecast','dailyforecast','min','max']]
    
    st.markdown('#### Shortage (pareto_items)')
    final_forcast=cleaned_pivot.merge(stock,left_on=['BranchCode','Name'],right_on=['Branch','MaterialName'],how='left')
    
    merged=final_forcast[final_forcast['Name'].isin(pareto_items['Name'])]
    merged[merged['UnRestrictedStock']<.5*merged['min']][['BranchCode','Name','min','UnRestrictedStock']]

    st.markdown('#### OVERSTOCK')
    final_forcast[final_forcast['UnRestrictedStock']>1.5*final_forcast['max']][['BranchCode','Name','max','UnRestrictedStock']]
    


    

    st.button("⬅️ Back to Main Menu", on_click=set_page, args=('home',))

    
