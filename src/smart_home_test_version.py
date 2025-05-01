import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

import os
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

def main():
    st.set_page_config(layout="wide", page_title="Phân tích điện năng")
    st.title("📊 Bộ công cụ phân tích điện năng")
    
    data = load_data()
    if data is None:
        return
    
    if 'use [kW]' not in data.columns:
        st.error("Không tìm thấy cột 'use [kW]' trong dữ liệu")
        st.write("Các cột số có sẵn:", data.columns.tolist())
        return
    
    st.sidebar.header("Tùy chọn hiển thị")
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("Đến ngày", max_date, min_value=min_date, max_value=max_date)
    
    if start_date > end_date:
        st.error("Ngày kết thúc phải sau ngày bắt đầu!")
        return
    
    try:
        filtered = data.loc[f"{start_date}":f"{end_date}"]
        # Chỉ lấy cột số để tính toán
        filtered = filtered.select_dtypes(include=['number'])
        daily_energy = calculate_daily(filtered)
        
        tab1, tab2 = st.tabs(["BIỂU ĐỒ THEO NGÀY", "TỔNG HỢP THEO NGÀY"])
        
        with tab1:
            # Lấy ngày có dữ liệu hợp lệ
            valid_dates = pd.Series(filtered.index.date).unique()
            
            if len(valid_dates) == 0:
                st.warning("Không có dữ liệu trong khoảng thời gian đã chọn")
                return
                
            selected_date = st.selectbox(
                "Chọn ngày để xem chi tiết",
                options=valid_dates,
                format_func=lambda x: x.strftime("%d/%m/%Y")
            )
            
            hourly_data = filtered[filtered.index.date == selected_date]
            
            if not hourly_data.empty:
                fig1 = px.area(
                    hourly_data, 
                    x=hourly_data.index, 
                    y='use [kW]',
                    title=f"Diễn biến công suất ngày {selected_date.strftime('%d/%m/%Y')}",
                    labels={'use [kW]': 'Công suất (kW)'}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                daily_kWh = hourly_data['use [kW]'].sum() / 60
                st.metric("Tổng tiêu thụ", f"{daily_kWh:.2f} kWh")
            else:
                st.warning("Không có dữ liệu cho ngày được chọn")
        
        with tab2:
            if not daily_energy.empty:
                fig2 = px.bar(
                    daily_energy,
                    x=daily_energy.index,
                    y='use [kW]',
                    title=f"Tổng năng lượng tiêu thụ từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}",
                    labels={'use [kW]': 'Năng lượng (kWh)'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                total = daily_energy['use [kW]'].sum()
                avg = daily_energy['use [kW]'].mean()
                
                cols = st.columns(3)
                cols[0].metric("Tổng năng lượng", f"{total:.2f} kWh")
                cols[1].metric("Trung bình ngày", f"{avg:.2f} kWh")
                cols[2].metric("Số ngày", len(daily_energy))
            else:
                st.warning("Không có dữ liệu trong khoảng thời gian này")
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")

if __name__ == "__main__":
    main()