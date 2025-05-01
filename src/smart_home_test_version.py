import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Load data
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

# Tải dữ liệu
data = load_data()

# Kiểm tra dữ liệu
if data is not None:
    # Hiển thị thông tin cơ bản
    st.title("⚡ Energy Data Dashboard")
    st.write("First 5 rows of data:", data.head())
    
    # Kiểm tra cột 'use [kW]' có tồn tại không
    if 'use [kW]' in data.columns:
        # Vẽ biểu đồ
        st.subheader("Energy Consumption Over Time")
        
        # Lấy 1000 điểm dữ liệu đầu tiên để hiển thị nhanh
        chart_data = data.head(1000).reset_index()
        
        fig = px.line(chart_data, 
                     x='datetime', 
                     y='use [kW]',
                     title='Energy Usage',
                     labels={'use [kW]': 'Power (kW)', 'datetime': 'Time'})
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Không tìm thấy cột 'use [kW]' trong dữ liệu")
        st.write("Các cột có sẵn:", data.columns.tolist())
else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn file.")