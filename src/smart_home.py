import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import datetime

# ====================== CẤU HÌNH ỨNG DỤNG ======================
st.set_page_config(
    page_title="Bảng Điều Khiển Năng Lượng Thông Minh",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

# ====================== KHAI BÁO HẰNG SỐ ======================
THIET_BI = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
    'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
    'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
    'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'
]

THOI_TIET = [
    'temperature', 'humidity', 'windSpeed', 
    'windBearing', 'pressure', 'apparentTemperature',
    'dewPoint', 'precipProbability'
]

# ====================== TIỀN XỬ LÝ DỮ LIỆU ======================
@st.cache_data
@st.cache_data
def load_data():
    try:
        # Đường dẫn tới file dữ liệu (điều chỉnh cho phù hợp)
        file_path = os.path.join("data", "HomeC.csv")
        
        # Đọc dữ liệu
        data = pd.read_csv(file_path, low_memory=False)
        
        # Xử lý dữ liệu
        data = data[:-1]  # Xóa dòng cuối nếu có NaN
        
        # Chuyển đổi cột time - xử lý lỗi nếu có
        if 'time' in data.columns:
            try:
                # Thử chuyển đổi từ Unix timestamp
                data['datetime'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
                
                # Nếu không thành công, thử chuyển đổi trực tiếp
                if data['datetime'].isnull().any():
                    data['datetime'] = pd.to_datetime(data['time'], errors='coerce')
                
                # Đặt index là datetime
                data = data.set_index('datetime')
                data = data.sort_index()
                
            except Exception as e:
                st.error(f"Lỗi chuyển đổi thời gian: {str(e)}")
                # Tạo timeline mẫu nếu cần
                data['datetime'] = pd.date_range(start='2016-01-01', periods=len(data), freq='min')
                data = data.set_index('datetime')
        
        return data.dropna()
    
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
        return None
def calculate_daily(df, power_col='use [kW]'):
    if power_col not in df.columns:
        return pd.DataFrame()
    # Đảm bảo chỉ tính toán trên cột số
    return df[[power_col]].resample('D').sum() / 60  # kW -> kWh
def calculate_hourly(df, power_col='use [kW]'):
    """Tính tổng công suất theo từng giờ (kWh)"""
    if power_col not in df.columns:
        return pd.DataFrame()
    # Tính tổng theo giờ và chuyển từ kW sang kWh (tích phân công suất)
    return df[[power_col]].resample('H').sum() / 60  # kW * 1h = kWh
def calculate_hourly_for_gen(df, power_col='gen [kW]'):
    """Tính tổng công suất theo từng giờ (kWh)"""
    if power_col not in df.columns:
        return pd.DataFrame()
    # Tính tổng theo giờ và chuyển từ kW sang kWh (tích phân công suất)
    return df[[power_col]].resample('H').sum() / 60  # kW * 1h = kWh

@st.cache_data
def xu_ly_du_lieu(_df):
    """Xử lý và làm giàu dữ liệu"""
    if _df is None:
        return None
        
    df = _df.copy()
    
    try:
        # Thêm các đặc trưng thời gian
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_name'] = df.index.day_name()
        df['weekend'] = df['day_of_week'].isin([5, 6])
        
        # Tính toán năng lượng
        df['net_energy'] = df['use [kW]'] - df['gen [kW]']
        df['energy_ratio'] = np.where(
            df['use [kW]'] > 0,
            df['gen [kW]'] / df['use [kW]'],
            0
        )
        
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        return None

# ====================== THÀNH PHẦN GIAO DIỆN ======================
def loc_ngay(df, key):
    """Hiển thị bộ lọc ngày"""
    try:
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        return st.date_input(
            "Chọn khoảng thời gian",
            [min_date, max_date],
            key=key,
            min_value=min_date,
            max_value=max_date
        )
    except Exception as e:
        st.error(f"Lỗi hiển thị bộ lọc: {str(e)}")
        return None

def hien_thi_chi_so(df):
    """Hiển thị các chỉ số năng lượng quan trọng"""
    if df is None:
        return
        
    cols = st.columns(4)
    metrics = [
        ("Tổng tiêu thụ", 'use [kW]', "sum", "Tổng năng lượng đã sử dụng"),
        ("Tổng sản xuất", 'gen [kW]', "sum", "Tổng năng lượng tạo ra"),
        ("Năng lượng ròng", 'net_energy', "sum", "Năng lượng thực tế (dùng - tạo)"),
        ("Tự cung cấp", None, "ratio", "Phần trăm nhu cầu được tự đáp ứng")
    ]
    
    for i, (ten, cot, loai, giai_thich) in enumerate(metrics):
        with cols[i]:
            try:
                if loai == "sum":
                    gia_tri = df[cot].sum()
                    st.metric(ten, f"{gia_tri:,.0f} kW", help=giai_thich)
                elif loai == "ratio":
                    ty_le = (df['gen [kW]'].sum() / df['use [kW]'].sum() * 100 
                           if df['use [kW]'].sum() > 0 else 0)
                    st.metric(ten, f"{ty_le:.1f}%", help=giai_thich)
            except Exception as e:
                st.error(f"Lỗi tính toán {ten}: {str(e)}")

# ====================== TRANG BẢNG ĐIỀU KHIỂN ======================
def trang_tong_quan(data):
    """Trang tổng quan năng lượng"""
    st.header("🏠 Tổng Quan Năng Lượng")
    
    if data is None:
        st.warning("Không có dữ liệu")
        return
        
    if 'use [kW]' not in data.columns:
        st.error("Không tìm thấy cột 'use [kW]' trong dữ liệu")
        st.write("Các cột số có sẵn:", data.columns.tolist())
        return
    
    start_date = data.index.min().date()
    end_date = data.index.max().date()
    
    if start_date > end_date:
        st.error("Ngày kết thúc phải sau ngày bắt đầu!")
        return
    
    try:
        filtered = data.loc[f"{start_date}":f"{end_date}"]
        filtered = filtered.select_dtypes(include=['number'])
        daily_energy = calculate_daily(filtered)
        hourly_energy = calculate_hourly(filtered)           
        hourly_data_gen = calculate_hourly_for_gen(filtered)
        
        tab1, tab2 = st.tabs(["NĂNG LƯỢNG TIÊU THỤ VÀ SẢN XUẤT HẰNG NGÀY", "TỔNG HỢP NĂNG LƯỢNG TIÊU THỤ VÀ SẢN XUẤT THEO NGÀY"])
        
        with tab1:
            valid_dates = pd.Series(filtered.index.date).unique()
            
            if len(valid_dates) == 0:
                st.warning("Không có dữ liệu trong khoảng thời gian đã chọn")
                return
                
            selected_date = st.selectbox(
                "Chọn ngày để xem chi tiết",
                options=valid_dates,
                format_func=lambda x: x.strftime("%d/%m/%Y")
            )
            st.markdown("") 
            # 
            hourly_data = hourly_energy[hourly_energy.index.date == selected_date]

            
            # Tính toán các chỉ số
            daily_total = hourly_data['use [kW]'].sum()
            max_hour = hourly_data['use [kW]'].idxmax()
            max_value = hourly_data['use [kW]'].max()
            avg_value = hourly_data['use [kW]'].mean()

            # Hiển thị các chỉ số dưới dạng columns
            cols = st.columns(3)
            cols[0].metric("Tổng năng lượng tiêu thụ", f"{daily_total:.2f} kWh")
            cols[1].metric("Giờ cao điểm", max_hour.strftime('%H:%M'), f"{max_value:.2f} kWh")
            cols[2].metric("Trung bình/giờ", f"{avg_value:.2f} kWh")
         
            st.markdown("---")   
            # Tính toán lại hourly_data_gen cho ngày đã chọn
            hourly_data_gen = hourly_data_gen[hourly_data_gen.index.date == selected_date]
            # Tính toán các chỉ số
            daily_total_gen = hourly_data_gen['gen [kW]'].sum()
            max_hour_gen = hourly_data_gen['gen [kW]'].idxmax()
            max_value_gen = hourly_data_gen['gen [kW]'].max()
            avg_value_gen = hourly_data_gen['gen [kW]'].mean()
            
            # Hiển thị các chỉ số dưới dạng columns
            cols = st.columns(3)
            cols[0].metric("Tổng năng lượng sản xuất", f"{daily_total_gen:.2f} kWh")
            cols[1].metric("Giờ cao điểm", max_hour_gen.strftime('%H:%M'), f"{max_value_gen:.2f} kWh")
            cols[2].metric("Trung bình/giờ", f"{avg_value_gen:.2f} kWh")
            
 
            # Hai biểu đồ này sẽ hiển thị năng lượng tiêu thụ và sản xuất theo giờ cho ngày đã chọn
            if not hourly_data.empty and not hourly_data_gen.empty:
                combined_data = hourly_data[['use [kW]']].join(hourly_data_gen[['gen [kW]']], how='outer').fillna(0)

                fig = px.line(
                    combined_data,
                    x=combined_data.index,
                    y=['use [kW]', 'gen [kW]'],
                    title=f"Năng lượng tiêu thụ và sản xuất theo giờ - Ngày {selected_date.strftime('%d/%m/%Y')}",
                    labels={'value': 'Năng lượng (kWh)', 'datetime': 'Giờ', 'variable': 'Loại năng lượng'},
                    markers=True
                )

                fig.update_traces(line=dict(width=3), marker=dict(size=8))

                fig.update_layout(
                    xaxis_tickformat='%H:%M',
                    hovermode="x unified",
                    yaxis_title="Năng lượng (kWh)",
                    xaxis_title="Thời gian",
                    legend_title_text='Loại năng lượng'
                )

                max_use_idx = combined_data['use [kW]'].idxmax()
                max_use_val = combined_data['use [kW]'].max()
                fig.add_annotation(
                    x=max_use_idx,
                    y=max_use_val,
                    text=f"Max tiêu thụ: {max_use_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )

                max_gen_idx = combined_data['gen [kW]'].idxmax()
                max_gen_val = combined_data['gen [kW]'].max()
                fig.add_annotation(
                    x=max_gen_idx,
                    y=max_gen_val,
                    text=f"Max sản xuất: {max_gen_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Không có dữ liệu cho ngày được chọn")

        
        with tab2:
          # Lấy phạm vi ngày có sẵn trong dữ liệu
            min_date = data.index.min().date()
            max_date = data.index.max().date()

            # Tạo giao diện chọn ngày
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Từ ngày", 
                                        min_date, 
                                        min_value=min_date, 
                                        max_value=max_date,
                                        key="start_date_selector")
            with col2:
                end_date = st.date_input("Đến ngày", 
                                    max_date, 
                                    min_value=min_date, 
                                    max_value=max_date,
                                    key="end_date_selector")

            # Kiểm tra hợp lệ ngày
            if start_date > end_date:
                st.error("Ngày kết thúc phải sau ngày bắt đầu!")
                st.stop()

            try:
                # Lọc dữ liệu theo khoảng ngày đã chọn
                date_mask = (data.index.date >= start_date) & (data.index.date <= end_date)
                filtered_data = data.loc[date_mask]
                
                # Tính toán năng lượng theo ngày
                daily_energy = filtered_data['use [kW]'].resample('D').sum() / 60  # Chuyển từ kW sang kWh
                
                if not daily_energy.empty:
                    # Vẽ biểu đồ cột
                    fig = px.bar(
                        daily_energy,
                        x=daily_energy.index,
                        y='use [kW]',
                        title=f"TỔNG NĂNG LƯỢNG TIÊU THỤ<br>Từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}",
                        labels={'use [kW]': 'Năng lượng (kWh)', 'index': 'Ngày'},
                        color_discrete_sequence=['#3498db']
                    )
                    
                    # Tùy chỉnh biểu đồ
                    fig.update_layout(
                        xaxis_tickformat='%d/%m',
                        hovermode="x unified",
                        plot_bgcolor='white',
                        height=450
                    )
                    
                    # Hiển thị giá trị trên mỗi cột
                    fig.update_traces(
                        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:.2f} kWh",
                        texttemplate='%{y:.1f}',
                        textposition='outside'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tính toán các chỉ số
                    total_energy = daily_energy.sum()
                    avg_energy = daily_energy.mean()
                    total_days = len(daily_energy)
                    
                    # Hiển thị thông số tổng hợp
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        label="TỔNG NĂNG LƯỢNG", 
                        value=f"{total_energy:,.2f} kWh",
                        delta=f"{total_days} ngày"
                    )
                    col2.metric(
                        label="TRUNG BÌNH NGÀY", 
                        value=f"{avg_energy:,.2f} kWh"
                    )
                    col3.metric(
                        label="HIỆU SUẤT CAO NHẤT", 
                        value=f"{daily_energy.max():.2f} kWh",
                        delta=f"Ngày {daily_energy.idxmax().strftime('%d/%m')}"
                    )
                    
                else:
                    st.warning(f"Không có dữ liệu từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}")
                    st.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")
                st.stop()
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {str(e)}")
        st.stop()


def trang_thiet_bi(df):
    """Trang phân tích theo thiết bị"""
    st.header("🔌 Phân Tích Theo Thiết Bị")
    
    if df is None:
        st.warning("Không có dữ liệu")
        return
        
    khoang_ngay = loc_ngay(df, "thiet_bi")
    df_loc = df[(df['date'] >= khoang_ngay[0]) & (df['date'] <= khoang_ngay[1])]
    
    thiet_bi_chon = st.multiselect(
        "Chọn thiết bị để phân tích",
        THIET_BI,
        default=THIET_BI[:3]
    )
    
    if not thiet_bi_chon:
        st.warning("Vui lòng chọn ít nhất một thiết bị")
        return
    
    st.markdown("---")
    
    try:
        tong_thiet_bi = df_loc[thiet_bi_chon].sum().sort_values(ascending=False)
        cols = st.columns(len(thiet_bi_chon))
        for i, (thiet_bi, tong) in enumerate(tong_thiet_bi.items()):
            with cols[i]:
                st.metric(
                    thiet_bi.replace(" [kW]", ""),
                    f"{tong:,.0f} kW",
                    help=f"Tổng tiêu thụ của {thiet_bi}"
                )
    except Exception as e:
        st.error(f"Lỗi tính toán tổng thiết bị: {str(e)}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Phân bổ", "⏱ Xu hướng", "🔗 Tương quan"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            try:
                fig = px.pie(
                    tong_thiet_bi,
                    values=tong_thiet_bi.values,
                    names=tong_thiet_bi.index.str.replace(" [kW]", ""),
                    title='Tỷ lệ tiêu thụ theo thiết bị'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị biểu đồ tròn: {str(e)}")
        
        with col2:
            try:
                fig = px.bar(
                    tong_thiet_bi.reset_index(),
                    x='index',
                    y=0,
                    title='Tổng tiêu thụ theo thiết bị',
                    labels={'index': 'Thiết bị', '0': 'Năng lượng (kW)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị biểu đồ cột: {str(e)}")
    
    with tab2:
        try:
            fig = px.line(
                df_loc.set_index('datetime')[thiet_bi_chon].resample('D').mean().reset_index(),
                x='datetime',
                y=thiet_bi_chon,
                title='Xu hướng sử dụng hàng ngày',
                labels={'value': 'Công suất (kW)', 'datetime': 'Ngày'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi hiển thị xu hướng: {str(e)}")
    
    with tab3:
        try:
            fig = px.imshow(
                df_loc[thiet_bi_chon].corr(),
                text_auto=True,
                aspect="auto",
                title='Mối tương quan giữa các thiết bị',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi hiển thị ma trận tương quan: {str(e)}")

def trang_thoi_tiet(df):
    """Trang phân tích ảnh hưởng thời tiết"""
    st.header("🌤️ Ảnh Hưởng Thời Tiết")
    
    if df is None:
        st.warning("Không có dữ liệu")
        return
        
    khoang_ngay = loc_ngay(df, "thoi_tiet")
    df_loc = df[(df['date'] >= khoang_ngay[0]) & (df['date'] <= khoang_ngay[1])]
    
    cols = st.columns(4)
    cac_thong_so = [
        ('temperature', '🌡️ Nhiệt độ TB', '°C'),
        ('humidity', '💧 Độ ẩm TB', '%'),
        ('windSpeed', '🌬️ Tốc độ gió TB', ' km/h'),
        ('pressure', '⏲️ Áp suất TB', ' hPa')
    ]
    
    for i, (cot, ten, don_vi) in enumerate(cac_thong_so):
        with cols[i]:
            try:
                gia_tri_tb = df_loc[cot].mean()
                st.metric(ten, f"{gia_tri_tb:.1f}{don_vi}")
            except Exception as e:
                st.error(f"Lỗi tính toán {ten}: {str(e)}")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🌦 Xu hướng", "⚡ Mối quan hệ"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            try:
                fig = px.line(
                    df_loc.set_index('datetime')[['temperature', 'apparentTemperature']].resample('D').mean().reset_index(),
                    x='datetime',
                    y=['temperature', 'apparentTemperature'],
                    title='Xu hướng nhiệt độ',
                    labels={'value': 'Nhiệt độ (°C)', 'datetime': 'Ngày'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị nhiệt độ: {str(e)}")
        
        with col2:
            try:
                fig = px.line(
                    df_loc.set_index('datetime')[['humidity', 'dewPoint']].resample('D').mean().reset_index(),
                    x='datetime',
                    y=['humidity', 'dewPoint'],
                    title='Độ ẩm & Điểm sương',
                    labels={'value': 'Giá trị', 'datetime': 'Ngày'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị độ ẩm: {str(e)}")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            try:
                mau_df = df_loc.sample(min(1000, len(df_loc)))
                fig = px.scatter(
                    mau_df,
                    x='temperature',
                    y='use [kW]',
                    color='hour',
                    trendline="lowess",
                    title='Nhiệt độ vs Tiêu thụ',
                    labels={'temperature': 'Nhiệt độ (°C)', 'use [kW]': 'Công suất (kW)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị biểu đồ nhiệt độ: {str(e)}")
        
        with col2:
            try:
                fig = px.scatter(
                    df_loc.sample(min(1000, len(df_loc))),
                    x='humidity',
                    y='use [kW]',
                    color='temperature',
                    trendline="lowess",
                    title='Độ ẩm vs Tiêu thụ',
                    labels={'humidity': 'Độ ẩm (%)', 'use [kW]': 'Công suất (kW)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi hiển thị biểu đồ độ ẩm: {str(e)}")

# ====================== ỨNG DỤNG CHÍNH ======================
def main():
    # Tải dữ liệu
    with st.spinner("Đang tải dữ liệu..."):
        df = load_data()
    
    # Xử lý dữ liệu
    with st.spinner("Đang xử lý dữ liệu..."):
        df_xu_ly = xu_ly_du_lieu(df)
    
    # Thanh điều hướng
    with st.sidebar:
        st.title("🏠 Điều Hướng")
        
        
        trang = st.radio(
            "Chọn trang",
            ["🏠 Tổng quan", "🔌 Thiết bị", "🌤️ Thời tiết"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("**Tóm tắt dữ liệu**")
        
        if df_xu_ly is not None:
            try:
                st.metric("Khoảng thời gian", 
                         f"{df_xu_ly['date'].min().strftime('%d/%m/%Y')} đến "
                         f"{df_xu_ly['date'].max().strftime('%d/%m/%Y')}")
            except Exception as e:
                st.error(f"Lỗi hiển thị tóm tắt: {str(e)}")
        else:
            st.warning("Không có dữ liệu")
        
        st.markdown("---")

        
        if df_xu_ly is not None and st.button("Tạo mẫu dữ liệu"):
            try:
                mau = df_xu_ly.sample(min(1000, len(df_xu_ly)))
                csv = mau.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống CSV",
                    data=csv,
                    file_name="mau_du_lieu.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Lỗi tạo mẫu: {str(e)}")
    
    # Điều hướng trang
    if df_xu_ly is not None:
        if trang == "🏠 Tổng quan":
            trang_tong_quan(df_xu_ly)
        elif trang == "🔌 Thiết bị":
            trang_thiet_bi(df_xu_ly)
        elif trang == "🌤️ Thời tiết":
            trang_thoi_tiet(df_xu_ly)
    else:
        st.error("Không tải được dữ liệu. Vui lòng kiểm tra file dữ liệu và thử lại.")

if __name__ == "__main__":
    main()