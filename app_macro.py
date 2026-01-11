# app_macro_viewer.py
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 2025
@author: trieukimlanh
🌍 Ứng dụng xem dữ liệu vĩ mô World Bank + CSV + Google Sheet
streamlit run "/Users/trieukimlanh/Library/CloudStorage/GoogleDrive-lanhtk@hub.edu.vn/My Drive/Spyder/macro/app_macro.py"
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import wb
import numpy as np
import datetime as dt
import re


# ==========================================================
# ⚙️ CONFIG
# ==========================================================
st.set_page_config(page_title="🌍 Macro Data Viewer (World Bank)", layout="wide")
st.title("📊 Macro Data Explorer")
st.sidebar.header("🌎 Tùy chọn dữ liệu")

# ==========================================================
# 🧩 NGUỒN DỮ LIỆU
# ==========================================================
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:",
    ["World Bank", "Upload CSV", "Google Sheet", "Kết hợp (World Bank + CSV/GS)"],
)

df_all = None
all_data = {}

# ==========================================================
# 📁 HÀM HỖ TRỢ
# ==========================================================
def prepare_csv(df):
    date_col = next((c for c in df.columns if c.lower() in ["date", "year", "time"]), None)
    if date_col:
        col_data = df[date_col]
        if pd.api.types.is_integer_dtype(col_data) or pd.api.types.is_float_dtype(col_data):
            df["Date"] = pd.to_datetime(col_data.astype(int).astype(str) + "-01-01")
        else:
            df["Date"] = pd.to_datetime(col_data, errors="coerce")

        # Không lọc mất các cột khác — chỉ set index
        df = df.set_index("Date").sort_index()

        # Thử ép kiểu numeric cho các cột còn lại nếu có thể
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    else:
        st.warning("⚠️ Không tìm thấy cột thời gian (Date/Year/Time).")

    return df


def load_csv_files(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            df_prepared = prepare_csv(df)
            dfs.append(df_prepared)
        except Exception as e:
            st.warning(f"Lỗi đọc file {file.name}: {e}")
    if dfs:
        df_csv = pd.concat(dfs, axis=1).dropna(how="all").sort_index()
        st.sidebar.success("✅ Đọc CSV thành công!")
        return df_csv
    return None

def read_gsheet_precise(sheet_link):
    """Đọc Google Sheet dạng CSV, tự động nhận diện cột và dữ liệu dạng số/chữ."""
    try:
        # Chuẩn hoá link export
        base_id = None
        gid = "0"
        if "/d/" in sheet_link:
            parts = sheet_link.split("/d/")
            base_id = parts[1].split("/")[0]
            if "gid=" in sheet_link:
                gid = sheet_link.split("gid=")[-1].split("&")[0].split("#")[0]
        if not base_id:
            st.warning("⚠️ Không nhận diện được ID của Google Sheet.")
            return None

        csv_url = f"https://docs.google.com/spreadsheets/d/{base_id}/export?format=csv&gid={gid}"

        # Thử đọc CSV, nếu lỗi dấu thập phân thì fallback
        try:
            df = pd.read_csv(csv_url)
        except Exception:
            df = pd.read_csv(csv_url, decimal=",", thousands=".")

        # Làm sạch tên cột
        df.columns = [str(c).strip() for c in df.columns]

        # Chuyển % và dấu phẩy sang số nếu cần
        for col in df.columns:
            if df[col].dtype == object:
                series = df[col].astype(str).str.strip()
                
                # Nếu cột này có >70% giá trị là số (sau khi bỏ ký tự), thì xử lý như số
                numeric_like = series.str.replace(r"[^0-9,.\-%]", "", regex=True)
                ratio_numeric = numeric_like.apply(lambda x: x.replace(",", ".").replace("%", "").replace("-", "").replace(".", "").isdigit()).mean()
        
                if ratio_numeric > 0.7:
                    series = (
                        series
                        .str.replace("%", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .str.replace(" ", "", regex=False)   # chỉ xoá khi là dạng số
                    )
                    df[col] = pd.to_numeric(series, errors="ignore")
                else:
                    # Nếu là text, giữ nguyên khoảng trắng
                    df[col] = series


        df_prepared = prepare_csv(df)
        return df_prepared

    except Exception as e:
        st.warning(f"❌ Lỗi đọc Google Sheet: {e}")
        return None



def load_wb_data(countries, indicators, start_year, end_year):
    try:
        df_wb = wb.download(indicator=indicators, country=countries, start=start_year, end=end_year)
        df_wb = df_wb.reset_index()
        df_wb = df_wb.pivot(index="year", columns=["country"], values=indicators)
        if isinstance(df_wb.columns, pd.MultiIndex):
            df_wb.columns = [f"{col[1]}_{col[0]}" for col in df_wb.columns]
        else:
            df_wb.columns = [str(col) for col in df_wb.columns]
        df_wb.index = pd.to_datetime(df_wb.index, format="%Y", errors='coerce')
        df_wb = df_wb.sort_index()
        st.sidebar.success("✅ Tải dữ liệu World Bank thành công!")
        return df_wb
    except Exception as e:
        st.error(f"Lỗi tải World Bank: {e}")
        return None

# ==========================================================
# 📝 SIDEBAR TRA CỨU & NHẬP DỮ LIỆU
# ==========================================================
if data_source in ["World Bank", "Kết hợp (World Bank + CSV/GS)"]:
    st.sidebar.subheader("🌎 Tra cứu World Bank")

    # Tra cứu chỉ tiêu
    with st.sidebar.expander("🔎 Tra cứu chỉ tiêu (Indicator)"):
        search_term = st.text_input("Nhập từ khóa (VD: gdp, inflation, export...)", "gdp per capita")
        if st.button("Tìm chỉ tiêu", key="search_indicator"):
            try:
                results = wb.search(search_term)
                st.write(results[['id','name','sourceNote']].head(20))
                st.success(f"Tìm thấy {len(results)} chỉ tiêu liên quan.")
            except Exception as e:
                st.error(f"Lỗi tìm kiếm: {e}")
            st.info("Dùng cột `id` làm mã chỉ tiêu tải về (VD: NY.GDP.PCAP.CD).")

    # Nhập mã chỉ tiêu
    indicator_input = st.sidebar.text_area(
        "Mã chỉ tiêu (VD: NY.GDP.MKTP.KD.ZG, FP.CPI.TOTL.ZG):",
        "NY.GDP.MKTP.KD.ZG\nFP.CPI.TOTL.ZG"
    )
    st.session_state["indicator_input"] = indicator_input

    # Hiển thị mô tả chỉ tiêu đã nhập
    indicator_list = [ind.strip() for ind in indicator_input.splitlines() if ind.strip()]
    if indicator_list:
        try:
            all_indicators = wb.get_indicators()
            matched = all_indicators[all_indicators["id"].isin(indicator_list)]
            if not matched.empty:
                st.sidebar.subheader("📘 Mô tả chỉ tiêu")
                for _, row in matched.iterrows():
                    with st.sidebar.expander(f"{row['id']} — {row['name']}"):
                        if isinstance(row['sourceNote'], str) and row['sourceNote']:
                            st.write(row['sourceNote'])
                        else:
                            st.write("Không có mô tả thêm.")
            else:
                st.sidebar.info("Không tìm thấy mô tả cho mã nhập vào.")
        except Exception as e:
            st.sidebar.warning(f"Lỗi khi tra mô tả chỉ tiêu: {e}")

    # Nhập quốc gia
    country_input = st.sidebar.text_input("Mã quốc gia (VD: VN, US, CN):", "VN,US")
    st.session_state["country_input"] = country_input

    # Chọn khoảng thời gian
    start_year_date = st.sidebar.date_input("Từ ngày (World Bank)", dt.date(2000,1,1),
                                            min_value=dt.date(1960,1,1), max_value=dt.date.today())
    end_year_date = st.sidebar.date_input("Đến ngày (World Bank)", dt.date.today(),
                                          min_value=dt.date(1960,1,1), max_value=dt.date.today())
    st.session_state["start_year"] = start_year_date.year
    st.session_state["end_year"] = end_year_date.year

# CSV / Google Sheet upload
# ==========================================================
# 📥 NHẬP DỮ LIỆU CSV / GOOGLE SHEET
# ==========================================================
uploaded_files = []
gsheet_links = []

# Cho phép nhập CSV hoặc Google Sheet trong cả 3 trường hợp:
# "Upload CSV", "Google Sheet", "Kết hợp (World Bank + CSV/GS)"
if data_source in ["Upload CSV", "Google Sheet", "Kết hợp (World Bank + CSV/GS)"]:
    st.sidebar.subheader("📁 Dữ liệu chính")

    # Upload CSV (tùy chọn)
    if data_source in ["Upload CSV", "Kết hợp (World Bank + CSV/GS)"]:
        uploaded_files = st.sidebar.file_uploader(
            "📂 Upload file CSV (có cột Date/Year/Time)",
            type="csv",
            accept_multiple_files=True
        )

    # Nhập link Google Sheet (tùy chọn)
    # ---- Dữ liệu Google Sheet ----
    if data_source in ["Google Sheet", "Kết hợp (World Bank + CSV/GS)"]:
        st.sidebar.markdown("🌐 **Hoặc nhập link Google Sheet DỮ LIỆU** (mỗi link 1 dòng)")
        gsheet_data_input = st.sidebar.text_area(
            "🔗 Link Google Sheet dữ liệu:",
            placeholder="https://docs.google.com/spreadsheets/d/xxxxxx/edit#gid=0",
            key="gsheet_data_input_data"  # 👈 đổi key
        )
        gsheet_links = [link.strip() for link in gsheet_data_input.splitlines() if link.strip()]
    else:
        gsheet_links = []  # 👈 reset rõ ràng khi không dùng

    
    # ——— Ngăn cách rõ phần mô tả —
    st.sidebar.divider()


# ==========================================================
# 📁 NHẬP FILE MÔ TẢ (indicator & unit)
# ==========================================================
st.sidebar.subheader("📄 Dữ liệu mô tả")

# 1️⃣ Upload CSV mô tả
desc_file = st.sidebar.file_uploader("Chọn file CSV mô tả", type="csv")

# 2️⃣ Link Google Sheet mô tả
gs_desc_link = st.sidebar.text_input(
    "Hoặc dán link Google Sheet mô tả (1 link)",
    placeholder="https://docs.google.com/spreadsheets/d/xxxxxx/edit#gid=0",
    key="gsheet_data_input_desc"
)

unit_map = {}
df_desc = pd.DataFrame()

if desc_file:
    try:
        df_desc = pd.read_csv(desc_file)
        if 'indicator' in df_desc.columns and 'unit' in df_desc.columns:
            unit_map = dict(zip(df_desc['indicator'], df_desc['unit']))
        else:
            st.warning("⚠️ File mô tả phải có cột 'indicator' và 'unit'.")
    except Exception as e:
        st.warning(f"❌ Lỗi đọc file mô tả: {e}")

elif gs_desc_link.strip():
    try:
        df_desc = read_gsheet_precise(gs_desc_link)
        if 'indicator' in df_desc.columns and 'unit' in df_desc.columns:
            unit_map = dict(zip(df_desc['indicator'], df_desc['unit']))
        else:
            st.warning("⚠️ Google Sheet mô tả phải có cột 'indicator' và 'unit'.")
    except Exception as e:
        st.warning(f"❌ Lỗi đọc Google Sheet mô tả: {e}")

# ✅ Chỉ hiển thị bảng mô tả ở đây một lần
if not df_desc.empty:
    st.markdown("📊 Thông tin mô tả")
    st.dataframe(df_desc, height=180, use_container_width=True)


# mặc định
if "mode" not in st.session_state:
    st.session_state["mode"] = "MACRO"

st.sidebar.divider()
st.sidebar.subheader("📘 Chế độ hiển thị")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📊 Macro"):
        st.session_state["mode"] = "MACRO"

with col2:
    if st.button("📘 FP"):
        st.session_state["mode"] = "FP"

if st.session_state["mode"] == "MACRO":

    st.subheader("📊 Phân tích Kinh tế Vĩ mô")

    # ==========================================================
    # ▶️ TẢI & HIỂN THỊ DỮ LIỆU
    # ==========================================================
    if "macro_loaded" not in st.session_state:
        st.session_state["macro_loaded"] = False

    if st.button("🚀 Macro"):
        all_data = {}
        total_unit_map = {}
        total_name_map = {}

        # ----------------------------
        # 1️⃣ WORLD BANK
        # ----------------------------
        if data_source in ["World Bank", "Kết hợp (World Bank + CSV/GS)"]:
            countries = [c.strip().upper() for c in st.session_state.get("country_input", "VN,US").split(",")]
            indicators = [ind.strip() for ind in st.session_state.get("indicator_input", "").splitlines() if ind.strip()]
            start_year = st.session_state.get("start_year", 2000)
            end_year = st.session_state.get("end_year", 2024)
    
            try:
                # Tải dữ liệu WB
                df_wb = wb.download(indicator=indicators, country=countries, start=start_year, end=end_year)
                df_wb = df_wb.reset_index().pivot(index="year", columns=["country"], values=indicators)
                if isinstance(df_wb.columns, pd.MultiIndex):
                    df_wb.columns = [f"{col[1]}_{col[0]}" for col in df_wb.columns]
                else:
                    df_wb.columns = [str(col) for col in df_wb.columns]
                df_wb.index = pd.to_datetime(df_wb.index, format="%Y", errors='coerce')
                df_wb = df_wb.sort_index()
                st.sidebar.success("✅ Tải dữ liệu World Bank thành công!")
    
                # Lấy unit / name từ meta
                df_meta = wb.get_indicators()
                for ind in indicators:
                    row = df_meta[df_meta["id"] == ind]
                    if not row.empty:
                        name = row.iloc[0]["name"]
                        note = row.iloc[0]["sourceNote"]
                        # Suy đoán đơn vị
                        match = re.search(r"\((.*?)\)", name)
                        unit = match.group(1) if match else None
                        if not unit and isinstance(note, str):
                            match = re.search(r"\((.*?)\)", note)
                            if match:
                                unit = match.group(1)
                            elif "percent" in note.lower():
                                unit = "%"
                        # Áp map theo cột
                        for col in df_wb.columns:
                            if col.endswith(f"_{ind}"):
                                total_unit_map[col] = unit or ""
                                total_name_map[col] = name
                all_data["WorldBank"] = df_wb
    
            except Exception as e:
                st.error(f"❌ Lỗi tải World Bank: {e}")
    
        # ----------------------------
        # 2️⃣ CSV
        # ----------------------------
        if uploaded_files:
            df_csv = load_csv_files(uploaded_files)
            if df_csv is not None:
                all_data["CSV"] = df_csv
                # Lấy unit từ file mô tả nếu có
                for c in df_csv.columns:
                    if c in unit_map:
                        total_unit_map[c] = unit_map[c]
                    else:
                        total_unit_map[c] = ""
                    total_name_map[c] = c
    
        # ----------------------------
        # 3️⃣ GOOGLE SHEET DỮ LIỆU
        # ----------------------------
        for link in gsheet_links:
            if link.strip():
                try:
                    df_gs = read_gsheet_precise(link)
                    # Đặt tên riêng cho từng sheet theo ID + GID
                    sheet_id = re.search(r"/d/([^/]+)/", link).group(1)
                    gid_match = re.search(r"gid=(\d+)", link)
                    gid = gid_match.group(1) if gid_match else "0"
                    sheet_name = f"GS_{sheet_id}_{gid}"
        
                    all_data[sheet_name] = df_gs
                    for c in df_gs.columns:
                        total_unit_map[c] = unit_map.get(c, "")
                        total_name_map[c] = c
        
                except Exception as e:
                    st.warning(f"❌ Lỗi đọc Google Sheet dữ liệu: {e}")
    
    
        # ----------------------------
        # ❌ Không có dữ liệu
        # ----------------------------
        if not all_data:
            st.error("❌ Không có dữ liệu nào được tải.")
            st.stop()
    
        # ----------------------------
        # 🔀 GỘP DỮ LIỆU
        # ----------------------------
        common_cols = ['Date', 'date', 'year', 'time']
        merge_col = None
        for col in common_cols:
            if all(col in df.columns for df in all_data.values()):
                merge_col = col
                break
    
        if merge_col:
            from functools import reduce
            df_all = reduce(lambda left, right: pd.merge(left, right, on=merge_col, how='outer'), all_data.values())
            st.success(f"✅ Hợp nhất dữ liệu theo cột `{merge_col}`")
        else:
            # Đảm bảo index của từng DataFrame là duy nhất
            for k, v in all_data.items():
                if not v.index.is_unique:
                    v = v[~v.index.duplicated(keep='first')]
                    all_data[k] = v
            
            # Sau đó mới concat
            df_all = pd.concat(all_data.values(), axis=1, join='outer').dropna(how='all')
    
            st.info("ℹ️ Hiển thị dữ liệu đã tải lên")
    
        # ----------------------------
        # 💾 Lưu vào session
        # ----------------------------
        st.session_state["df_all"] = df_all
        st.session_state["unit_map"] = total_unit_map
        st.session_state["name_map"] = total_name_map
        st.session_state["macro_loaded"] = True
        # ----------------------------
        # 📋 Hiển thị dữ liệu
        # ----------------------------
        #st.dataframe(df_all)
        # Chuẩn hoá hiển thị cột thời gian
        df_display = df_all.copy()
        if isinstance(df_display.index, pd.DatetimeIndex):
            df_display.index = df_display.index.strftime("%Y-%m-%d")
        
        st.sidebar.success(f"✅ Tổng {len(df_all.columns)} cột dữ liệu đã tải.")
        
    # ==========================================================
    # 📊 HIỂN THỊ THEO TAB
    # ==========================================================
    if st.session_state.get("macro_loaded", False):
        tab_data, tab_desc, tab_chart = st.tabs([
            "📊 Dữ liệu",
            "🧾 Thống kê mô tả",
            "📈 Biểu đồ"
        ])
        df_display = st.session_state["df_all"].copy()
        if isinstance(df_display.index, pd.DatetimeIndex):
            df_display.index = df_display.index.strftime("%Y-%m-%d")
        # =========================
        # TAB 1 – DỮ LIỆU
        # =========================
        with tab_data:
            st.markdown("📊 Dữ liệu đã tải")
            st.dataframe(df_display, use_container_width=True)
            
        # =========================
        # TAB 2 – THỐNG KÊ MÔ TẢ
        # =========================
        with tab_desc:
            st.markdown("🧾 Thống kê mô tả")
            st.dataframe(df_display.describe(), use_container_width=True)
    
        # =========================
        # TAB 3 – BIỂU ĐỒ
        # =========================
        with tab_chart:
            # ==========================================================
            # 📊 VẼ DỮ LIỆU (có tuỳ chỉnh thời gian & tần suất)
            # ==========================================================
            if "df_all" in st.session_state:
                df_plot = st.session_state["df_all"]
            
                # ========== 🕒 Chọn khoảng thời gian ==========
                st.markdown("🕒 Lọc thời gian hiển thị")
                # Đảm bảo index có kiểu datetime
                if not isinstance(df_plot.index, pd.DatetimeIndex):
                    if "Date" in df_plot.columns:
                        df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce", format="%Y", exact=False)
                        df_plot = df_plot.set_index("Date")
                    else:
                        st.error("⚠️ Không tìm thấy cột 'Date' trong dữ liệu CSV.")
                        st.stop()
                
                # Nếu dữ liệu chỉ có năm, điền mặc định là ngày 1/1
                if df_plot.index.dtype == 'datetime64[ns]':
                    df_plot.index = df_plot.index.fillna(pd.Timestamp("1900-01-01"))
                
                # Loại bỏ NaT trong index
                df_plot = df_plot[~df_plot.index.isna()]
                
                if not df_plot.empty:
                    min_date = df_plot.index.min().date()
                    max_date = df_plot.index.max().date()
                else:
                    st.warning("⚠️ Không có dữ liệu hợp lệ sau khi xử lý cột thời gian.")
                    st.stop()
            
                if isinstance(df_plot.index, pd.DatetimeIndex):
                    min_date = df_plot.index.min().date()
                    max_date = df_plot.index.max().date()
            
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Từ ngày:", min_date, min_value=min_date, max_value=max_date)
                    with col2:
                        end_date = st.date_input("Đến ngày:", max_date, min_value=min_date, max_value=max_date)
            
                    # Lọc bỏ giá trị thời gian NaT trước khi so sánh
                    if isinstance(df_plot.index, pd.DatetimeIndex):
                        df_plot = df_plot[~df_plot.index.isna()]
                    
                        mask = (df_plot.index >= pd.to_datetime(start_date)) & (df_plot.index <= pd.to_datetime(end_date))
                        df_plot = df_plot.loc[mask]
            
                else:
                    st.info("Dữ liệu chưa có định dạng thời gian, hiển thị toàn bộ.")
            
                # ========== ⏳ Chọn tần suất ==========
                st.markdown("⏳ Chọn tần suất vẽ dữ liệu")
                freq_map = {
                    "Năm": "Y",
                    "Quý": "Q",
                    "Tháng": "M",
                    "Tuần": "W",
                    "Ngày": "D"
                }
                freq_choice = st.radio("Tần suất hiển thị:", list(freq_map.keys()), horizontal=True)
                freq_code = freq_map[freq_choice]
            
                if isinstance(df_plot.index, pd.DatetimeIndex):
                    # Giữ cột numeric riêng để tính
                    df_num = df_plot.select_dtypes(include="number")
                    
                    # Nếu có cột text, lưu lại để merge sau
                    df_nonnum = df_plot.select_dtypes(exclude="number")
                    
                    # Resample chỉ phần numeric
                    df_resampled = df_num.resample(freq_code).mean()
                    
                    # Gắn lại cột text nếu cần (giữ giá trị đầu tiên của mỗi giai đoạn)
                    if not df_nonnum.empty:
                        df_text = df_nonnum.resample(freq_code).first()
                        df_plot = pd.concat([df_resampled, df_text], axis=1)
                    else:
                        df_plot = df_resampled
            
            
                # ========== 🧩 Chọn cột để vẽ ==========
                st.markdown("🧩 Chọn chỉ tiêu hiển thị")
                # Tự động nhận diện cột có thể là numeric
                cols_to_plot = st.multiselect(
                    "Chọn cột dữ liệu để vẽ:",
                    options=df_plot.columns.tolist(),
                    default=df_plot.columns[:3].tolist() if len(df_plot.columns) >= 3 else df_plot.columns.tolist()
                )
            
                if cols_to_plot:
                    #df_subset = df_plot[cols_to_plot].dropna(how="all")
                    df_subset = df_plot[cols_to_plot].dropna(how="any")
                    fig, ax = plt.subplots(figsize=(12, 5))
            
                    # Phân trục dựa theo đơn vị (nếu có)
                    unit_map = st.session_state.get("unit_map", {})
                    name_map = st.session_state.get("name_map", {})
                    left_cols, right_cols = [], []
                    left_names, right_names = {}, {}
            
                    if unit_map:
                        units = [unit_map.get(col, None) for col in cols_to_plot]
                        unique_units = list(set([u for u in units if u]))
            
                        if len(unique_units) > 1:
                            left_unit = unique_units[0]
                            for col in cols_to_plot:
                                name_label = name_map.get(col, col)
                                if unit_map.get(col) == left_unit:
                                    left_cols.append(col)
                                    left_names[col] = name_label
                                else:
                                    right_cols.append(col)
                                    right_names[col] = name_label
                        else:
                            left_cols = cols_to_plot
                            left_names = {col: name_map.get(col, col) for col in left_cols}
                    else:
                        left_cols = cols_to_plot
                        left_names = {col: col for col in left_cols}
            
                    # ======== Vẽ trục trái ========
                    colors_left = plt.cm.Set2(np.linspace(0, 1, len(left_cols)))
                    for i, col in enumerate(left_cols):
                        ax.plot(
                            df_subset.index, df_subset[col],
                            label=left_names[col],
                            color=colors_left[i],
                            linewidth=2
                        )
                    #ax.set_ylabel(", ".join([left_names[c] for c in left_cols]), color="tab:blue")
                    ax.tick_params(axis='y', labelcolor="tab:blue")
                    ax.set_xlabel("Thời gian")
                    ax.grid(alpha=0.3)
                    
                    # ======== Vẽ trục phải (nếu có) ========
                    if right_cols:
                        ax2 = ax.twinx()
                        colors_right = plt.cm.Set1(np.linspace(0, 1, len(right_cols)))
                        for i, col in enumerate(right_cols):
                            ax2.plot(
                                df_subset.index, df_subset[col],
                                linestyle='--',
                                label=right_names[col],
                                color=colors_right[i],
                                linewidth=2
                            )
                        #ax2.set_ylabel(", ".join([right_names[c] for c in right_cols]), color="tab:red")
                        ax2.tick_params(axis='y', labelcolor="tab:red")
                    
                        # Gộp legend của cả 2 trục
                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
                    else:
                        ax.legend(fontsize=8)
            
            
                    # ======== Tiêu đề biểu đồ ========
                    title_parts = []
                    if left_cols:
                        title_parts.append(", ".join([left_names[c] for c in left_cols]))
                    if right_cols:
                        title_parts.append(", ".join([right_names[c] for c in right_cols]))
                    ax.set_title(" & ".join(title_parts), fontsize=14)
            
                    st.pyplot(fig)
            
                    # ========== 💾 Tải dữ liệu lọc ra ==========
                    csv_export = df_subset.reset_index().to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Tải dữ liệu hiển thị (CSV)",
                        csv_export,
                        file_name="filtered_plot_data.csv",
                        mime="text/csv"
                    )

# =========== LẬP TRÌNH TÀI CHÍNH ===========
st.sidebar.subheader("📥 Dữ liệu Lập trình tài chính (IMF)")

# File uploader
fp_file = st.sidebar.file_uploader(
    "📂 Upload CSV FP (wide format, có cột sector)",
    type="csv"
)

# Đường dẫn mặc định nếu không upload
#default_fp_file = '/Users/trieukimlanh/Library/CloudStorage/OneDrive-Personal/0. Research Projects onedrive/Macroeconomic reports/Tập 16/Vietnam FPP.csv'
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
default_fp_file = BASE_DIR / "Vietnam FPP.csv"

# Nút tải dữ liệu
if st.sidebar.button("🚀 Tải dữ liệu FP"):

    # Nếu user upload file thì dùng file upload, nếu không thì dùng đường dẫn mặc định
    fp_file_to_load = fp_file if fp_file is not None else default_fp_file

    # Đọc CSV
    df_fp = pd.read_csv(fp_file_to_load)

    # ---- kiểm tra cột bắt buộc ----
    required_cols = {"sector", "indicator", "unit"}
    if not required_cols.issubset(df_fp.columns):
        st.sidebar.error("❌ File phải có các cột: sector, indicator, unit.")
        st.stop()

    # ---- kiểm tra sector ----
    valid_sectors = {1, 2, 3, 4}
    if not set(df_fp["sector"].dropna().unique()).issubset(valid_sectors):
        st.sidebar.error("❌ sector chỉ được nhận giá trị 1–4.")
        st.stop()

    # ---- tách theo sector (giữ dạng wide) ----
    fp_data = {
        1: df_fp[df_fp["sector"] == 1],
        2: df_fp[df_fp["sector"] == 2],
        3: df_fp[df_fp["sector"] == 3],
        4: df_fp[df_fp["sector"] == 4],
    }

    # ---- lưu vào session_state để các tab khác dùng ----
    st.session_state["fp_data"] = fp_data
    st.session_state["unit_map"] = dict(zip(df_fp["indicator"], df_fp["unit"]))
    st.session_state["fp_loaded"] = True

    st.sidebar.success("✅ Đã tải dữ liệu FP thành công!")

# Nếu muốn load trực tiếp từ file mặc định (không cần bấm nút)
if "fp_data" not in st.session_state:
    df_fp = pd.read_csv(default_fp_file)
    st.session_state["fp_data"] = {
        1: df_fp[df_fp["sector"] == 1],
        2: df_fp[df_fp["sector"] == 2],
        3: df_fp[df_fp["sector"] == 3],
        4: df_fp[df_fp["sector"] == 4],
    }
    st.session_state["unit_map"] = dict(zip(df_fp["indicator"], df_fp["unit"]))
    st.session_state["fp_loaded"] = True

# =========================
# LOAD DATA CHO CÁC TAB
# =========================
if st.session_state["mode"] == "FP":
    st.subheader("📘 Macroeconomic Diagnosis (IMF Style)")
    st.caption("Diagnostic & Narrative: Real – External – Fiscal – Monetary")

    df_real = st.session_state["fp_data"][1]
    df_ext = st.session_state["fp_data"][2]
    df_fiscal = st.session_state["fp_data"][3]
    df_monetary = st.session_state["fp_data"][4]

    tab_real, tab_ext, tab_fiscal, tab_monetary, tab_fpp = st.tabs([
        "🏭 Real",
        "🌍 External",
        "🏛 Fiscal",
        "💰 Monetary",
        "🔗 FPP"
    ])

    
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    def render_fp_tab(df, sector_name, description):
        st.subheader(sector_name)
        st.markdown(description)
    
        if df.empty:
            st.info("Không có dữ liệu.")
            return
    
        year_cols = [c for c in df.columns if c.isdigit()]
        indicators = df["indicator"].tolist()


        selected = st.multiselect(
            "Chọn chỉ tiêu:",
            indicators,
            key=sector_name
        )
        if not selected:
            st.info("Hãy chọn ít nhất một chỉ tiêu để hiển thị.")
            return
        unit_map = {ind: st.session_state["unit_map"][ind] for ind in selected}

        groups = defaultdict(list)
        for ind in selected:
            groups[unit_map[ind]].append(ind)
        
        # =========================
        # BUILD BASE DATA (CHUNG)
        # =========================
        st.markdown("### 📊 Bảng dữ liệu")
        filtered_df = df[df["indicator"].isin(selected)]
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("### ⚙️ Chart options")

        chart_mode = st.radio(
            "Chế độ vẽ",
            ["📌 Theo từng chỉ tiêu (drop NA)", "📅 So sánh theo giai đoạn chung"],
            horizontal=True,
            key=f"chart_mode_{sector_name}"
        )
   
        year_cols_int = sorted([int(c) for c in year_cols])
        
        min_year, max_year = min(year_cols_int), max(year_cols_int)
        
        year_range = st.slider(
            "📆 Chọn khoảng năm hiển thị:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
            key=f"year_range_{sector_name}"
        )
        
        start_year, end_year = year_range
        selected_years = [str(y) for y in range(start_year, end_year + 1)]
        selected_years = [y for y in selected_years if y in year_cols]

        st.markdown("### 📈 Diễn biến theo thời gian")

        plot_df = (
            df[df["indicator"].isin(selected)]
            .set_index("indicator")[selected_years]
            .T
            .astype(float)
        )
        
        if chart_mode == "📌 Theo từng chỉ tiêu (drop NA)":
            # giữ sample riêng cho từng chỉ tiêu
            plot_df = plot_df.dropna(how="all")
        else:
            # chỉ giữ những năm mà tất cả chỉ tiêu đều có dữ liệu
            plot_df = plot_df.dropna()

        # =========================
        # CASE 1: 1 UNIT
        # =========================
        if len(groups) == 1:
            fig, ax = plt.subplots(figsize=(5, 2))  # hoặc (6,3), (8,4) tùy bạn

            for ind in selected:
                s = plot_df[ind].dropna()
                ax.plot(s.index, s.values, label=ind)
    
            unit = list(groups.keys())[0]
            
            ax.yaxis.get_offset_text().set_fontsize(5)
            ax.grid(True)
            #ax.set_xlabel("Year", fontsize=5)
            #ax.set_ylabel(unit, fontsize=5)
            ax.tick_params(axis='both', labelsize=5)
            ax.legend(fontsize=5)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
        # =========================
        # CASE 2: 2 UNITS (DUAL AXIS)
        # =========================
        elif len(groups) == 2:
            fig, ax1 = plt.subplots(figsize=(5, 2))  # hoặc (6,3), (8,4) tùy bạn
            plt.xticks(rotation=45)

            units = list(groups.keys())
            left_unit, right_unit = units[0], units[1]
    
            for ind in groups[left_unit]:
                s = plot_df[ind].dropna()
                ax1.plot(s.index, s.values, label=ind)
    
            #ax1.set_xlabel("Year", fontsize=8)
            #ax1.set_ylabel(left_unit, fontsize=5)
            ax1.tick_params(axis='both', labelsize=5)
            ax1.legend(fontsize=5) 
            ax1.yaxis.get_offset_text().set_fontsize(5)

            
            ax2 = ax1.twinx()
            for ind in groups[right_unit]:
                s = plot_df[ind].dropna()
                ax2.plot(s.index, s.values, linestyle="--", label=ind)
    
            #ax2.set_ylabel(right_unit, fontsize=5)
            ax2.tick_params(axis='y', labelsize=5)
            ax2.yaxis.get_offset_text().set_fontsize(5)

            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=5)
            ax1.grid(True)

            st.pyplot(fig)
    
        # =========================
        # CASE 3: >2 UNITS → MULTIPLE CHARTS
        # =========================
        else:
            st.info("Có nhiều hơn 2 đơn vị đo → tách thành nhiều biểu đồ.")
    
            for unit, inds in groups.items():
                fig, ax = plt.subplots(figsize=(5, 2))  # hoặc (6,3), (8,4) tùy bạn

                for ind in inds:
                    s = plot_df[ind].dropna()
                    ax.plot(s.index, s.values, label=ind)
                
                ax.yaxis.get_offset_text().set_fontsize(5)
                ax.grid(True)
                #ax.set_xlabel("Year", fontsize=5)
                #ax.set_ylabel(unit, fontsize=5)
                ax.tick_params(axis='both', labelsize=5)
                ax.legend(fontsize=5)
                plt.xticks(rotation=45)
                st.pyplot(fig)


        
        def fmt(x, unit=""):
            if pd.isna(x):
                return "NA"
            if abs(x) >= 1000:
                return f"{x:,.1f}{unit}"
            else:
                return f"{x:.1f}{unit}"

        
        # =========================
        # DIAGNOSTIC BLOCK (IMF STYLE)
        # =========================
        st.markdown("### 🧠 Diagnostic signals")
        
        def get_sector_of_indicator(df, ind):
            return df[df["indicator"] == ind]["sector"].iloc[0]
        
        def classify_indicator(ind):
            name = ind.lower()
        
            if "gdp" in name and "percent change" in name:
                return "gdp_growth"
            if "unemployment" in name:
                return "unemployment"
            if "cpi" in name and "percent change" in name:
                return "inflation"
            if "current account" in name:
                return "current_account"
            if "exports" in name or "imports" in name:
                return "trade"
            if "gross debt" in name:
                return "public_debt"
            if "net lending" in name or "balance" in name:
                return "fiscal_balance"
            if "revenue" in name:
                return "revenue"
            if "expenditure" in name:
                return "expenditure"
            if "m2" in name or "money" in name:
                return "money"
            if "credit" in name:
                return "credit"
            if "interest rate" in name or "policy rate" in name:
                return "policy_rate"
        
            return "other"

        for ind in selected:
        
            s = plot_df[ind].dropna()
            if len(s) < 2:
                continue
        
            latest_year = s.index.max()
            latest = s.loc[latest_year]
            prev = s.shift(1).loc[latest_year]
            delta = latest - prev
        
            sector = get_sector_of_indicator(df, ind)
            itype = classify_indicator(ind)
        
            label = f"**{ind}** ({latest_year})"
        
            # ================= REAL SECTOR =================
            if sector == 1:
        
                if itype == "gdp_growth":
                    if latest > 5:
                        st.success(f"📈 {label}: tăng trưởng mạnh ở mức {fmt(latest, '%')}.")
                    elif latest < 2:
                        st.warning(f"📉 {label}: tăng trưởng yếu ({fmt(latest, '%')}).")
                    else:
                        st.info(f"ℹ️ {label}: tăng trưởng trung bình ({fmt(latest, '%')}).")
        
                elif itype == "unemployment":
                    if delta > 0:
                        st.warning(f"⚠️ {label}: thất nghiệp tăng lên {fmt(latest, '%')}.")
                    else:
                        st.success(f"✅ {label}: thất nghiệp giảm xuống {fmt(latest, '%')}.")
                
                else:
                    st.info(f"ℹ️ {label}: {fmt(latest)} (Δ {fmt(delta)})")
        
            # ================= EXTERNAL SECTOR =================
            elif sector == 2:
        
                if itype == "current_account":
                    if latest < 0:
                        st.warning(f"⚠️ {label}: thâm hụt vãng lai ({fmt(latest, '%')}).")
                    else:
                        st.success(f"✅ {label}: thặng dư vãng lai ({fmt(latest, '%')}).")
        
                elif itype == "trade":
                    if latest > 0:
                        st.success(f"📈 {label}: tăng trưởng tích cực ({fmt(latest, '%')}).")
                    else:
                        st.warning(f"📉 {label}: suy giảm ({fmt(latest, '%')}).")
        
                else:
                    st.info(f"ℹ️ {label}: {fmt(latest)} (Δ {fmt(delta)})")
        
            # ================= FISCAL SECTOR =================
            elif sector == 3:
        
                if itype == "public_debt":
                    if latest > 60:
                        st.warning(f"⚠️ {label}: nợ công cao ({fmt(latest, '%')} GDP).")
                    else:
                        st.success(f"✅ {label}: nợ công trong ngưỡng an toàn ({fmt(latest, '%')} GDP).")
        
                elif itype == "fiscal_balance":
                    if latest < 0:
                        st.warning(f"⚠️ {label}: thâm hụt ngân sách ({fmt(latest, '%')} GDP).")
                    else:
                        st.success(f"✅ {label}: thặng dư ngân sách ({fmt(latest, '%')} GDP).")
        
                else:
                    st.info(f"ℹ️ {label}: {fmt(latest)} (Δ {fmt(delta)})")
        
            # ================= MONETARY SECTOR =================
            elif sector == 4:
        
                if itype == "inflation":
                    if latest > 4:
                        st.warning(f"🔥 {label}: lạm phát cao ({fmt(latest, '%')}).")
                    else:
                        st.success(f"✅ {label}: lạm phát được kiểm soát ({fmt(latest, '%')}).")
        
                elif itype == "money":
                    if delta > 0:
                        st.info(f"💰 {label}: cung tiền mở rộng ({fmt(latest)}).")
                    else:
                        st.warning(f"⚠️ {label}: cung tiền thu hẹp.")
        
                elif itype == "credit":
                    if delta > 0:
                        st.success(f"📈 {label}: tín dụng tăng ({fmt(latest)}).")
                    else:
                        st.warning(f"📉 {label}: tín dụng chậm lại.")
        
                elif itype == "policy_rate":
                    if delta > 0:
                        st.warning(f"⚠️ {label}: thắt chặt tiền tệ ({fmt(latest, '%')}).")
                    else:
                        st.info(f"ℹ️ {label}: nới lỏng tiền tệ ({fmt(latest, '%')}).")
        
                else:
                    st.info(f"ℹ️ {label}: {fmt(latest)} (Δ {fmt(delta)})")


        
    #=====TAB1: REAL SECTOR===========
    with tab_real:
        render_fp_tab(
            df_real,
            "🏭 Real Sector",
            """
            **Diagnostic focus**:
            - Tăng trưởng có bền vững không?
            - Lạm phát có gây áp lực chính sách?
            """
        )


    
    #======TAB2: EXTERNAL SECTOR=========
    with tab_ext:
        render_fp_tab(
            df_ext,
            "🌍 External Sector",
            """
            **Diagnostic focus**:
            - Cán cân vãng lai có bền vững?
            - Rủi ro thanh khoản đối ngoại?
            """
        )


    
    #======TAB3: FISCAL SECTOR========
    with tab_fiscal:
        render_fp_tab(
            df_fiscal,
            "🏛 Fiscal Sector",
            """
            **Diagnostic focus**:
            - Thâm hụt có vượt ngưỡng?
            - Nợ công có xu hướng tăng nhanh?
            """
        )


    
    #=======TAB4: MONETARY SECTOR=========
    with tab_monetary:
        render_fp_tab(
            df_monetary,
            "💰 Monetary Sector",
            """
            **Diagnostic focus**:
            - Tín dụng có tăng nóng?
            - Chính sách tiền tệ có đủ chặt?
            """
        )
   
    #========TAB5: FPP============  
    with tab_fpp:
        # ---- chuẩn hoá sang long ----
        def to_long(df):
            year_cols = [c for c in df.columns if c.isdigit()]
            return df.melt(
                id_vars=["sector", "indicator", "unit"],
                value_vars=year_cols,
                var_name="year",
                value_name="value"
            ).dropna()
    
        df_long = pd.concat(
            [
                to_long(df_real),
                to_long(df_ext),
                to_long(df_fiscal),
                to_long(df_monetary),
            ],
            axis=0
        )
        # =========================
        # Helper functions (GIỮ NGUYÊN LOGIC BẠN)
        # =========================
        def find_indicator(df, keyword_list):
            for ind in df["indicator"]:
                name = ind.lower()
                if all(k.lower() in name for k in keyword_list):
                    return ind
            return None
    
        def get_series(df, indicator):
            if indicator is None:
                return None
            year_cols = [c for c in df.columns if c.isdigit()]
            s = (
                df[df["indicator"] == indicator]
                .set_index("indicator")[year_cols]
                .T
                .astype(float)
                .iloc[:, 0]
            )
            s.index = s.index.astype(int)
            return s.sort_index()
        
        # =========================
        # HÀM CHUNG VẼ ĐỒ THỊ THEO UNIT
        # =========================
        import matplotlib.pyplot as plt
        import streamlit as st
        
        def plot_fpp_chart(plot_df, unit_map=None, display_to_original=None, title="🔗 FPP Chart"):
            """
            plot_df: DataFrame, index là years, columns là series
            unit_map: dict, key=series_name, value=unit từ dữ liệu gốc
            title: tiêu đề chart
            """
            
            if unit_map is None:
                # Lấy từ session_state nếu đã load
                unit_map = st.session_state.get("unit_map", {})
                
            if plot_df.empty:
                st.warning("No data to plot.")
                return
            
            # =========================
            # Helper: lấy unit
            # =========================
            if display_to_original is None:
                display_to_original = {}
            def get_unit(col):
                original_name = display_to_original.get(col, col)
                return unit_map.get(original_name, "Other")
        
            # =========================
            # Tạo groups tự động: unit -> list series
            # =========================
            from collections import defaultdict

            groups = defaultdict(list)
            for col in plot_df.columns:
                groups[get_unit(col)].append(col)
    
            from matplotlib.ticker import MaxNLocator
            # =========================
            # CASE 1: 1 UNIT
            # =========================
            if len(groups) == 1:
                fig, ax = plt.subplots(figsize=(5, 2))
                for unit, inds in groups.items():
                    for ind in inds:
                        s = plot_df[ind].dropna()
                        ax.plot(s.index, s.values, label=ind)
        
                ax.yaxis.get_offset_text().set_fontsize(5)
                ax.grid(True)
                ax.tick_params(axis='both', labelsize=5)
                ax.legend(fontsize=5)
                ax.set_title(title, fontsize=6)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
            # =========================
            # CASE 2: 2 UNITS (DUAL AXIS)
            # =========================
            elif len(groups) == 2:
                fig, ax1 = plt.subplots(figsize=(5, 2))
                plt.xticks(rotation=45)
        
                units = list(groups.keys())
                left_unit, right_unit = units[0], units[1]
        
                # Trục trái
                for ind in groups[left_unit]:
                    if ind in plot_df.columns:
                        s = plot_df[ind].dropna()
                        ax1.plot(s.index, s.values, label=ind)
        
                ax1.tick_params(axis='both', labelsize=5)
                ax1.yaxis.get_offset_text().set_fontsize(5)
        
                # Trục phải
                ax2 = ax1.twinx()
                for ind in groups[right_unit]:
                    if ind in plot_df.columns:
                        s = plot_df[ind].dropna()
                        ax2.plot(s.index, s.values, linestyle="--", label=ind)
        
                ax2.tick_params(axis='y', labelsize=5)
                ax2.yaxis.get_offset_text().set_fontsize(5)
        
                # Kết hợp legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=5)
                ax1.grid(True)
                ax1.set_title(title, fontsize=6)
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                st.pyplot(fig)
        
            # =========================
            # CASE 3: >2 UNITS → MULTIPLE CHARTS
            # =========================
            else:
                st.info("Có nhiều hơn 2 đơn vị đo → tách thành nhiều biểu đồ.")
                for unit, inds in groups.items():
                    fig, ax = plt.subplots(figsize=(5, 2))
                    for ind in inds:
                        if ind in plot_df.columns:
                            s = plot_df[ind].dropna()
                            ax.plot(s.index, s.values, label=ind)
        
                    ax.yaxis.get_offset_text().set_fontsize(5)
                    ax.grid(True)
                    ax.tick_params(axis='both', labelsize=5)
                    ax.legend(fontsize=5)
                    ax.set_title(f"{title} ({unit})", fontsize=6)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
    
        # =========================
        # 3. BUSINESS CYCLE BLOCK (IMF STYLE)
        # =========================
        st.markdown("#### 1️⃣ Business Cycle & Macro Stance")
        
        #st.markdown("##### 📐 IMF interpretation framework")
        #st.latex(r"\text{Output gap} = \frac{Y - Y^*}{Y^*}")
        #st.latex(r"\pi_t = \frac{CPI_t - CPI_{t-1}}{CPI_{t-1}}")
        real_inds = df_real["indicator"].tolist()
        gdp_growth_ind = st.selectbox(
            "GDP growth (real, %)",
            real_inds,key="bc_gdp_growth",
            index=real_inds.index(find_indicator(df_real, ["gdp", "constant", "percent change"]))
            if find_indicator(df_real, ["gdp", "constant", "percent change"]) in real_inds else 0
        )
    
        inflation_ind = st.selectbox(
            "Inflation (CPI, %)",
            real_inds,key="bc_inflation",
            index=real_inds.index(find_indicator(df_real, ["cpi", "percent change"]))
            if find_indicator(df_real, ["cpi", "period average", "percent change"]) in real_inds else 0
        )
    
        output_gap_ind = st.selectbox(
            "Output gap (PF)",
            real_inds,key="bc_output",
            index=real_inds.index(find_indicator(df_real, ["output gap_pf"]))
            if find_indicator(df_real, ["output gap_pf"]) in real_inds else 0
        )
        # ---- Tạo name_map ngay block ----
        display_to_original = {
            "GDP growth (%)": gdp_growth_ind,
            "Inflation (%)": inflation_ind,
            "Output gap (%)": output_gap_ind
        }
        
        #---- Lấy series ----
        bc_series = {}
        for display_name, indicator_var in display_to_original.items():
            series = get_series(df_real, indicator_var)
            if series is not None:
                bc_series[display_name] = series
        
        # ---- Align years ----
        common_years = None
        for s in bc_series.values():
            common_years = s.index if common_years is None else common_years.intersection(s.index)
        
        bc_df = pd.DataFrame({
            name: s.loc[common_years] for name, s in bc_series.items()
        }).dropna()
        bc_df.index = bc_df.index.astype(int)
        
        # ---- Raw table ----
        st.markdown("##### 📊 Raw table")
        st.dataframe(bc_df.round(2), use_container_width=True)
    
        # ---- Chart options ----
        st.markdown("##### ⚙️ Chart options")
    
        bc_cols = st.multiselect(
            "Select series to plot",
            bc_df.columns.tolist(),
            default=bc_df.columns.tolist(),
            key="bc_plot_cols"
        )
    
        min_year, max_year = int(bc_df.index.min()), int(bc_df.index.max())
    
        year_range = st.slider(
            "Select year range",
            min_year,
            max_year,
            (min_year, max_year),
            key="bc_year_range"
        )
    
        # ---- Filter ----
        plot_df = bc_df.loc[
            (bc_df.index >= year_range[0]) &
            (bc_df.index <= year_range[1]),
            bc_cols
        ]
    
        # ---- Plot ----
        st.markdown("##### 📈 Plots")
        
        if not plot_df.empty:
            plot_fpp_chart(
                plot_df,
                unit_map=st.session_state.get("unit_map", {}),
                display_to_original=display_to_original,
                title="Business cycle indicators"
            )

    
        # ---- Diagnostics ----
        st.markdown("##### 🧠 Diagnostics")
    
        last = bc_df.iloc[-1]
    
        if "Output gap (%)" in last and last["Output gap (%)"] > 0:
            st.info("Economy above potential: positive output gap.")
    
        if "Output gap (%)" in last and last["Output gap (%)"] < 0:
            st.warning("Economy below potential: negative output gap.")
    
        if "Inflation (%)" in last and "GDP growth (%)" in last:
            if last["Inflation (%)"] > 5 and last["GDP growth (%)"] < 2:
                st.warning("Stagflation risk: high inflation with weak growth.")
    
        

        # =========================
        # 4. FLOW OF FUNDS (IMF CORE IDENTITY)
        # =========================
        st.markdown("#### 2️⃣ Flow of Funds")
        
        st.markdown("##### 📐 Accounting identity")
        st.latex(r"(S - I)_{private} + (T - G) + (M - X) = 0")
        st.latex(r"\Rightarrow Private\ Balance = CAB - Fiscal\ Balance")
        ext_inds = df_ext["indicator"].tolist()
        cab_ind = st.selectbox(
            "Current Account Balance (% GDP)",
            ext_inds,key="ff_CAB",
            index=ext_inds.index(find_indicator(df_ext, ["current account", "percent of gdp"]))
            if find_indicator(df_ext, ["current account", "percent of gdp"]) in ext_inds else 0
        )
        fiscal_inds = df_fiscal["indicator"].tolist()
    
        fiscal_balance_ind = st.selectbox(
            "Fiscal balance (% GDP)",
            fiscal_inds,key="ff_FB",
            index=fiscal_inds.index("Net lending (+) / net borrowing (-), General government, Percent of GDP")
            if "Net lending (+) / net borrowing (-), General government, Percent of GDP" in fiscal_inds else 0
        )
        
        # ---- Mapping cục bộ cho block ----
        display_to_original = {
            "CAB (% GDP)": cab_ind,
            "Fiscal balance (% GDP)": fiscal_balance_ind,
        }
        
        # ---- Extract series ----
        ff_series = {}
        for display_name, indicator_var in display_to_original.items():
            if display_name == "CAB (% GDP)":
                s = get_series(df_ext, indicator_var)
            else:
                s = get_series(df_fiscal, indicator_var)
        
            if s is not None:
                ff_series[display_name] = s
        
        # ---- Kiểm tra đủ series ----
        if len(ff_series) == 2:
        
            # ---- Align years ----
            common_years = None
            for s in ff_series.values():
                common_years = s.index if common_years is None else common_years.intersection(s.index)
        
            flow_df = pd.DataFrame({
                name: s.loc[common_years]
                for name, s in ff_series.items()
            }).dropna()
        
            # ---- Residual ----
            flow_df["Private balance (residual)"] = (
                flow_df["CAB (% GDP)"] - flow_df["Fiscal balance (% GDP)"]
            )
        
            flow_df.index = flow_df.index.astype(int)
        
            # ---- Raw table ----
            st.markdown("##### 📊 Raw table")
            st.dataframe(flow_df.round(2), use_container_width=True)
        
            # ---- Chart options ----
            st.markdown("##### ⚙️ Chart options")
        
            cols = flow_df.columns.tolist()
        
            ff_cols = st.multiselect(
                "Select series to plot",
                cols,
                default=cols,
                key="ff_plot_cols"
            )
        
            min_year, max_year = int(flow_df.index.min()), int(flow_df.index.max())
        
            year_range = st.slider(
                "Select year range",
                min_year,
                max_year,
                (min_year, max_year),
                key="ff_year_range"
            )
        
            # ---- Filter ----
            plot_df = flow_df.loc[
                (flow_df.index >= year_range[0]) &
                (flow_df.index <= year_range[1]),
                ff_cols
            ]
        
            # ---- Plot ----
            st.markdown("##### 📈 Plots")
        
            if not plot_df.empty:
                plot_fpp_chart(
                    plot_df,
                    unit_map=st.session_state.get("unit_map", {}),
                    display_to_original=display_to_original,
                    title="Flow of Funds balances"
                )

            # ---- Diagnostics ----
            st.markdown("##### 🧠 Diagnostics")
        
            last = flow_df.iloc[-1]
        
            if last["Fiscal balance (% GDP)"] < 0 and last["CAB (% GDP)"] < 0:
                st.warning("Twin deficits: both fiscal and external balances are negative.")
        
            if abs(last["Private balance (residual)"]) > 5:
                st.info("Large private sector imbalance detected.")

    
        # =========================
        # 5. DEBT DYNAMICS (IMF STYLE)
        # =========================
        st.markdown("#### 3️⃣ Debt Dynamics")
        
        st.markdown("##### 📐 Standard IMF equation")
        st.latex(r"b_t = \frac{1+r_t}{1+g_t} b_{t-1} - pb_t")
        st.latex(r"\Delta b_t \approx (r_t - g_t)b_{t-1} - pb_t")
        interp_debt_df = pd.DataFrame({
            "Ký hiệu": ["b_t", "b_{t-1}", "r_t", "g_t", 
                        "pb_t", "(r_t - g_t)b_{t-1}", "-pb_t"],
            "Ý nghĩa": [
                "Nợ công hiện tại (% GDP)",
                "Nợ công năm trước (% GDP)",
                "Lãi suất thực của nợ công",
                "Tăng trưởng GDP thực",
                "Cán cân ngân sách cơ bản (% GDP)",
                "Snowball effect (hiệu ứng lăn tuyết)",
                "Fiscal effect (tác động tài khóa)"
            ],
            "Diễn giải kinh tế": [
                "Quy mô gánh nặng nợ của nền kinh tế",
                "Cơ sở để nợ tăng/giảm trong năm nay",
                "Chi phí vay nợ của chính phủ",
                "Khả năng tăng thu nhập để trả nợ",
                "Kỷ luật tài khóa hay kích thích tài khóa",
                "Áp lực vĩ mô làm nợ phình to nếu r > g",
                "Nếu pb dương → giảm nợ; nếu âm → tăng nợ"
            ]
        })
        
        st.markdown("###### 📘 Diễn giải phương trình nợ công")
        st.dataframe(interp_debt_df, use_container_width=True)

        
        # ---- Use mapped IMF variables ----
        fiscal_inds = df_fiscal["indicator"].tolist()
        real_inds = df_real["indicator"].tolist()
        mon_inds = df_monetary["indicator"].tolist()
        primary_balance_ind = st.selectbox(
            "Primary balance (% GDP)",
            fiscal_inds,key="dd_pb",
            index=fiscal_inds.index(find_indicator(df_fiscal, ["primary", "percent of gdp"]))
            if find_indicator(df_fiscal, ["primary", "percent of gdp"]) in fiscal_inds else 0
        )
    
        debt_ind = st.selectbox(
            "Public debt (% GDP)",
            fiscal_inds,key="dd_public",
            index=fiscal_inds.index(find_indicator(df_fiscal, ["gross debt", "percent of gdp"]))
            if find_indicator(df_fiscal, ["gross debt", "percent of gdp"]) in fiscal_inds else 0
        )
        
        gdp_growth_ind = st.selectbox(
            "GDP growth (real, %)",
            real_inds,key="dd_gdpg",
            index=real_inds.index(find_indicator(df_real, ["gdp", "constant", "percent change"]))
            if find_indicator(df_real, ["gdp", "constant", "percent change"]) in real_inds else 0
        )
        policy_rate_ind = st.selectbox(
            "Policy rate (%)",
            mon_inds,key="dd_rate",
            index=mon_inds.index(find_indicator(df_monetary, ["policy", "rate"]))
            if find_indicator(df_monetary, ["policy", "rate"]) in mon_inds else 0
        )
        
        # ---- Local mapping ----
        display_to_original = {
            "Debt (% GDP)": debt_ind,
            "Primary balance (% GDP)": primary_balance_ind,
            "GDP growth (%)": gdp_growth_ind,
            "Policy rate (%)": policy_rate_ind
        }

        
        # ---- Extract series ----
        dd_series = {}
        
        debt_s = get_series(df_fiscal, debt_ind)
        pb_s = get_series(df_fiscal, primary_balance_ind)
        growth_s = get_series(df_real, gdp_growth_ind)
        rate_s = get_series(df_monetary, policy_rate_ind)

        if rate_s is not None:
            dd_series["Policy rate (%)"] = rate_s

        if debt_s is not None:
            dd_series["Debt (% GDP)"] = debt_s
        
        if pb_s is not None:
            dd_series["Primary balance (% GDP)"] = pb_s
        
        if growth_s is not None:
            dd_series["GDP growth (%)"] = growth_s
        
        if len(dd_series) > 0:
        
            # ---- Align years ----
            common_years = None
            for s in dd_series.values():
                common_years = s.index if common_years is None else common_years.intersection(s.index)
        
            dd_df = pd.DataFrame({
                name: s.loc[common_years] for name, s in dd_series.items()
            }).dropna()
        
            dd_df.index = dd_df.index.astype(int)
        
            # ---- Decomposition (approximate) ----
            if all(x in dd_df.columns for x in ["Debt (% GDP)", "GDP growth (%)", "Policy rate (%)"]):
                dd_df["b(t-1)"] = dd_df["Debt (% GDP)"].shift(1)
                dd_df["g"] = dd_df["GDP growth (%)"] / 100
                dd_df["r"] = dd_df["Policy rate (%)"] / 100
                
                dd_df["Snowball effect"] = (dd_df["r"] - dd_df["g"]) * dd_df["b(t-1)"]

            # Fiscal effect and snowball effect
            if "Primary balance (% GDP)" in dd_df:
                dd_df["Fiscal effect"] = - dd_df["Primary balance (% GDP)"]
            
            if "Snowball effect" in dd_df and "Fiscal effect" in dd_df:
                dd_df["Δ Debt"] = dd_df["Snowball effect"] + dd_df["Fiscal effect"]
            
            # Sau khi tính snowball và fiscal effect
            dd_df = dd_df.dropna(subset=["Snowball effect", "Fiscal effect", "Δ Debt"])

            # ---- Raw table ----
            st.markdown("##### 📊 Raw table")
            st.dataframe(dd_df, use_container_width=True)
                        
            # ---- Chart options ----
            st.markdown("##### ⚙️ Chart options")
            # vẽ đồ thị decomposition
            min_year_d, max_year_d = int(dd_df.index.min()), int(dd_df.index.max())

            debt_decomp_year_range = st.slider(
                "Select year range (ΔDebt decomposition)",
                min_year_d,
                max_year_d,
                (min_year_d, max_year_d),
                key="debt_decomp_year_range"
            )
            
            plot_debt_decomp_df = dd_df.loc[
                (dd_df.index >= debt_decomp_year_range[0]) &
                (dd_df.index <= debt_decomp_year_range[1])
            ]

            def plot_debt_decomposition(decomp_df, title="ΔDebt decomposition"):
                fig, ax = plt.subplots(figsize=(5, 2))
            
                ax.bar(
                    decomp_df.index,
                    decomp_df["Snowball effect"],
                    label="Snowball effect"
                )
            
                ax.bar(
                    decomp_df.index,
                    decomp_df["Fiscal effect"],
                    bottom=decomp_df["Snowball effect"],
                    label="Fiscal effect"
                )
            
                ax.plot(
                    decomp_df.index,
                    decomp_df["Δ Debt"],
                    linestyle="--",
                    label="ΔDebt (approx)"
                )
            
                ax.yaxis.get_offset_text().set_fontsize(5)
                ax.legend(fontsize=5)
                ax.grid(True)
                ax.tick_params(axis='both', labelsize=5)
                ax.set_title(title, fontsize=6)
            
                st.pyplot(fig)
                
            if not plot_debt_decomp_df.empty:
                plot_debt_decomposition(plot_debt_decomp_df, title="ΔDebt = Snowball + Fiscal")

            # vẽ đồ thị các chỉ tiêu        
            dd_cols = st.multiselect(
                "Select series to plot",
                dd_df.columns.tolist(),
                default=["Debt (% GDP)"],
                key="dd_plot_cols"
            )
        
            min_year, max_year = int(dd_df.index.min()), int(dd_df.index.max())
        
            year_range = st.slider(
                "Select year range",
                min_year,
                max_year,
                (min_year, max_year),
                key="dd_year_range"
            )
        
            # ---- Filter ----
            plot_df = dd_df.loc[
                (dd_df.index >= year_range[0]) &
                (dd_df.index <= year_range[1]),
                dd_cols
            ]
        
            # ---- Plot ----
            st.markdown("##### 📈 Plots")
        
            if not plot_df.empty:
                
                plot_fpp_chart(
                    plot_df,
                    unit_map=st.session_state.get("unit_map", {}),
                    display_to_original=display_to_original,
                    title="Debt dynamics"
                )
                      
            # ---- Diagnostics ----
            st.markdown("##### 🧠 Diagnostics")
        
            last = dd_df.iloc[-1]
        
            if "Debt (% GDP)" in last and last["Debt (% GDP)"] > 60:
                st.warning("High public debt level.")
        
            if "Δ Debt" in last and last["Δ Debt"] > 3:
                st.info("Debt is rising rapidly.")


        # PART 3 — Monetary Conditions + Risk Narrative Engine
        # =========================
        # 6. MONETARY CONDITIONS (IMF STYLE)
        # =========================
        st.markdown("#### 4️⃣ Monetary Conditions")
        
        st.markdown("##### 📐 Conceptual framework")
        #st.latex(r"i_t - \pi_t \approx r_t")
        st.latex(r"\Delta M = \Delta NFA + \Delta NDA")
        
        # ---- Pickers ----
        mon_inds = df_monetary["indicator"].tolist()
        
        policy_rate_ind = st.selectbox(
            "Policy rate (%)",
            mon_inds,key="mon_rate",
            index=mon_inds.index(find_indicator(df_monetary, ["policy", "rate"]))
            if find_indicator(df_monetary, ["policy", "rate"]) in mon_inds else 0
        )
        
        credit_ind = st.selectbox(
            "Credit",
            mon_inds,key="mon_credit",
            index=mon_inds.index(find_indicator(df_monetary, ["credit"]))
            if find_indicator(df_monetary, ["credit"]) in mon_inds else 0
        )
        
        m2_ind = st.selectbox(
            "Money supply (M2)",
            mon_inds,key="mon_m2",
            index=mon_inds.index(find_indicator(df_monetary, ["m2"]))
            if find_indicator(df_monetary, ["m2"]) in mon_inds else 0
        )
        
        nfa_ind = st.selectbox(
            "Net Foreign Assets (NFA)",
            mon_inds,
            key="mon_nfa",
            index=mon_inds.index(find_indicator(df_monetary, ["nfa"]))
            if find_indicator(df_monetary, ["nfa"]) in mon_inds else 0
        )
        
        nda_ind = st.selectbox(
            "Net Domestic Assets (NDA)",
            mon_inds,
            key="mon_nda",
            index=mon_inds.index(find_indicator(df_monetary, ["nda"]))
            if find_indicator(df_monetary, ["nda"]) in mon_inds else 0
        )
        
        # ---- Local mapping ----
        display_to_original = {
            "Policy rate": policy_rate_ind,
            "Credit": credit_ind,
            "Money (M2)": m2_ind,
            "NFA": nfa_ind,
            "NDA": nda_ind
        }    
        # ---- Extract series ----
        mon_series = {}
        
        pr_s = get_series(df_monetary, policy_rate_ind)
        cr_s = get_series(df_monetary, credit_ind)
        m2_s = get_series(df_monetary, m2_ind)
        nfa_s = get_series(df_monetary, nfa_ind)
        nda_s = get_series(df_monetary, nda_ind)
        
        if nfa_s is not None:
            mon_series["NFA"] = nfa_s
        if nda_s is not None:
            mon_series["NDA"] = nda_s
        if pr_s is not None:
            mon_series["Policy rate"] = pr_s
        if cr_s is not None:
            mon_series["Credit"] = cr_s
        if m2_s is not None:
            mon_series["Money (M2)"] = m2_s
        
        if len(mon_series) > 0:
        
            # ---- Align years ----
            common_years = None
            for s in mon_series.values():
                common_years = s.index if common_years is None else common_years.intersection(s.index)
        
            mon_df = pd.DataFrame({
                name: s.loc[common_years]
                for name, s in mon_series.items()
            }).dropna()
        
            mon_df.index = mon_df.index.astype(int)
            # ---- Raw table ----
            
            st.markdown("##### 📊 Raw table")
            st.dataframe(mon_df.round(2), use_container_width=True)
            
            delta_df = mon_df.diff().dropna()
            if all(x in delta_df.columns for x in ["Money (M2)", "NFA", "NDA"]):
                decomp_df = pd.DataFrame({
                    "ΔM2": delta_df["Money (M2)"],
                    "ΔNFA": delta_df["NFA"],
                    "ΔNDA": delta_df["NDA"]
                })
        
                decomp_df["Residual"] = decomp_df["ΔM2"] - (decomp_df["ΔNFA"] + decomp_df["ΔNDA"])
            st.markdown("##### 🔍 Monetary decomposition: ΔM = ΔNFA + ΔNDA")
            st.dataframe(decomp_df.round(2), use_container_width=True)
            
            # ---- Decomposition plot ----
            st.markdown("##### 📉 Decomposition plot")
            st.markdown("###### 🧭 Tiền tăng là do đâu? (Interpretation guide)")

            interp_df = pd.DataFrame({
                "Trường hợp": [
                    "ΔM tăng vì ΔNDA",
                    "ΔM tăng vì ΔNFA",
                    "ΔM giảm vì ΔNDA âm",
                    "ΔM giảm vì ΔNFA âm",
                ],
                "Hàm ý": [
                    "Kích cầu nội địa, tín dụng, tài khóa",
                    "Dòng vốn ngoại vào, can thiệp FX",
                    "Thắt chặt tiền tệ trong nước",
                    "Chảy vốn ra / áp lực cán cân ngoại",
                ]
            })
            
            st.dataframe(interp_df, use_container_width=True)

            min_year_d, max_year_d = int(decomp_df.index.min()), int(decomp_df.index.max())
            
            decomp_year_range = st.slider(
                "Select year range (ΔM decomposition)",
                min_year_d,
                max_year_d,
                (min_year_d, max_year_d),
                key="mon_decomp_year_range"
            )
            
            plot_decomp_df = decomp_df.loc[
                (decomp_df.index >= decomp_year_range[0]) &
                (decomp_df.index <= decomp_year_range[1])
            ]
            
            def plot_monetary_decomposition(decomp_df, title="ΔM = ΔNFA + ΔNDA"):
                fig, ax = plt.subplots(figsize=(5, 2))
            
                ax.bar(decomp_df.index, decomp_df["ΔNFA"], label="ΔNFA")
                ax.bar(decomp_df.index, decomp_df["ΔNDA"], bottom=decomp_df["ΔNFA"], label="ΔNDA")
            
                ax.plot(decomp_df.index, decomp_df["ΔM2"], linestyle="--", label="ΔM2")
                ax.yaxis.get_offset_text().set_fontsize(5)
                ax.legend(fontsize=5)
                ax.grid(True)
                ax.tick_params(axis='both', labelsize=5)
                ax.set_title(title, fontsize=6)
                #plt.xticks(rotation=45)
            
                st.pyplot(fig)
            
            if not plot_decomp_df.empty:
                plot_monetary_decomposition(plot_decomp_df)

            
            # ---- Chart options ----
            st.markdown("##### ⚙️ Chart options")
            mon_cols = st.multiselect(
                "Select series to plot",
                mon_df.columns.tolist(),
                default=mon_df.columns.tolist(),
                key="mon_plot_cols"
            )
        
            min_year, max_year = int(mon_df.index.min()), int(mon_df.index.max())
            
            year_range = st.slider(
                "Select year range",
                min_year,
                max_year,
                (min_year, max_year),
                key="mon_year_range"
            )
        
            # ---- Filter ----
            plot_df = mon_df.loc[
                (mon_df.index >= year_range[0]) &
                (mon_df.index <= year_range[1]),
                mon_cols
            ]
        
            # ---- Plot ----
            st.markdown("##### 📈 Plots")
        
            if not plot_df.empty:
                plot_fpp_chart(
                    plot_df,
                    unit_map=st.session_state.get("unit_map", {}),
                    display_to_original=display_to_original,
                    title="Monetary conditions"
                )

            # ---- Diagnostics ----
            st.markdown("##### 🧠 Diagnostics")
            
            last = mon_df.iloc[-1]
            prev = mon_df.iloc[-2] if len(mon_df) >= 2 else None
            
            # ---- Credit dynamics ----
            if "Credit" in last and prev is not None:
                if last["Credit"] > prev["Credit"]:
                    st.success("Credit expansion detected.")
                else:
                    st.warning("Credit growth slowing.")
            
            # ---- Policy stance ----
            if "Policy rate" in last and prev is not None:
                if last["Policy rate"] > prev["Policy rate"]:
                    st.warning("Monetary tightening.")
                else:
                    st.info("Monetary easing.")
            
            # ---- Monetary source diagnostics: ΔM = ΔNFA + ΔNDA ----
            if all(x in mon_df.columns for x in ["Money (M2)", "NFA", "NDA"]) and len(mon_df) >= 2:
                delta = mon_df.diff().iloc[-1]
            
                dM = delta["Money (M2)"]
                dNFA = delta["NFA"]
                dNDA = delta["NDA"]
            
                # Direction of monetary conditions
                if dM > 0:
                    st.success("Net monetary expansion.")
                else:
                    st.warning("Net monetary contraction.")
            
                # Source of money creation
                if abs(dNDA) > abs(dNFA):
                    if dNDA > 0:
                        st.info("Money growth is mainly driven by domestic credit expansion (ΔNDA).")
                    else:
                        st.warning("Domestic deleveraging is the main drag on money supply (ΔNDA).")
                else:
                    if dNFA > 0:
                        st.info("Money growth is mainly driven by foreign inflows / reserve accumulation (ΔNFA).")
                    else:
                        st.warning("Capital outflows or FX interventions are draining liquidity (ΔNFA).")
            
                # Consistency check
                residual = dM - (dNFA + dNDA)
                if abs(residual) > 0.05 * abs(dM):
                    st.warning("⚠️ Monetary identity shows large residual — possible data inconsistency.")

   
        # =========================
        # 7. IMF-STYLE RISK NARRATIVE ENGINE
        # =========================
        st.markdown("#### 5️⃣ IMF-style Risk Narrative")
        
        # ---- Pickers ----
        st.markdown("##### 🎯 Variable selection")
        
        real_inds = df_real["indicator"].tolist()
        fiscal_inds = df_fiscal["indicator"].tolist()
        ext_inds = df_ext["indicator"].tolist()
        
        inflation_ind = st.selectbox(
            "Inflation (CPI, %)",
            real_inds,key="nar_inflation",
            index=real_inds.index(find_indicator(df_real, ["cpi", "percent change"]))
            if find_indicator(df_real, ["cpi", "period average", "percent change"]) in real_inds else 0
        )
    
        output_gap_ind = st.selectbox(
            "Output gap (PF)",
            real_inds,key="nar_output",
            index=real_inds.index(find_indicator(df_real, ["output gap_pf"]))
            if find_indicator(df_real, ["output gap_pf"]) in real_inds else 0
        )
        
        debt_ind = st.selectbox(
            "Public debt (% GDP)",
            fiscal_inds,key="nar_public",
            index=fiscal_inds.index(find_indicator(df_fiscal, ["gross debt", "percent of gdp"]))
            if find_indicator(df_fiscal, ["gross debt", "percent of gdp"]) in fiscal_inds else 0
        )
        
        cab_ind = st.selectbox(
            "Current account (% GDP)",
            ext_inds,key="nar_CA",
            index=ext_inds.index(find_indicator(df_ext, ["current account", "percent of gdp"]))
            if find_indicator(df_ext, ["current account", "percent of gdp"]) in ext_inds else 0
        )
        
        # ---- Extract series ----
        series_dict = {}
        
        og_s = get_series(df_real, output_gap_ind)
        inf_s = get_series(df_real, inflation_ind)
        debt_s = get_series(df_fiscal, debt_ind)
        cab_s = get_series(df_ext, cab_ind)
        
        if og_s is not None:
            series_dict["Output gap"] = og_s
        if inf_s is not None:
            series_dict["Inflation"] = inf_s
        if debt_s is not None:
            series_dict["Public debt"] = debt_s
        if cab_s is not None:
            series_dict["Current account"] = cab_s
        
        if len(series_dict) >= 1:
        
            # ---- Align years ----
            common_years = None
            for s in series_dict.values():
                common_years = s.index if common_years is None else common_years.intersection(s.index)
        
            nar_df = pd.DataFrame({
                name: s.loc[common_years]
                for name, s in series_dict.items()
            }).dropna()
        
            nar_df.index = nar_df.index.astype(int)
        
            # ---- Raw table ----
            st.markdown("##### 📊 Raw table")
            st.dataframe(nar_df.round(2), use_container_width=True)
        
            # ---- Chart options ----
            st.markdown("##### ⚙️ Chart options")
        
            nar_cols = st.multiselect(
                "Select indicators for narrative engine",
                nar_df.columns.tolist(),
                default=nar_df.columns.tolist(),
                key="nar_plot_cols"
            )
        
            min_year, max_year = int(nar_df.index.min()), int(nar_df.index.max())
        
            year_range = st.slider(
                "Select year range",
                min_year,
                max_year,
                (min_year, max_year),
                key="nar_year_range"
            )
        
            plot_df = nar_df.loc[
                (nar_df.index >= year_range[0]) &
                (nar_df.index <= year_range[1]),
                nar_cols
            ]
        
            # ---- Plot ----
            st.markdown("##### 📈 Plots")
        
            if not plot_df.empty:
                plot_fpp_chart(
                    plot_df,
                    unit_map=st.session_state.get("unit_map", {}),
                    display_to_original=display_to_original,
                    title="Risk-relevant macro indicators"
                )
                
            # ---- IMF-style narrative rules ----
            st.markdown("##### 🧠 Narrative")
        
            last = nar_df.iloc[-1]
            narrative = []
        
            if "Output gap" in last:
                if last["Output gap"] > 0:
                    narrative.append("The economy is operating above potential, indicating demand-side pressures.")
                else:
                    narrative.append("The economy is operating below potential, suggesting economic slack.")
        
            if "Inflation" in last:
                if last["Inflation"] > 4:
                    narrative.append("Inflationary pressures remain elevated.")
                else:
                    narrative.append("Inflation appears contained.")
        
            if "Public debt" in last:
                if last["Public debt"] > 60:
                    narrative.append("Public debt is elevated, raising sustainability concerns.")
        
            if "Current account" in last:
                if last["Current account"] < 0:
                    narrative.append("The external position shows a current account deficit.")
        
            if len(narrative) == 0:
                st.info("Not enough information to generate narrative.")
            else:
                st.write(" ".join(narrative))

        # =========================
        # 8. BASELINE vs ALTERNATIVE SCENARIOS
        # =========================
        st.markdown("#### 6️⃣ Baseline vs Alternative Scenario")
        
        if len(series_dict) >= 1:
        
            scenario_var = st.selectbox(
                "Select variable for scenario analysis",
                nar_df.columns.tolist(),
                key="scenario_var"
            )
        
            baseline = nar_df[scenario_var]
        
            st.markdown("##### ⚙️ Alternative scenario assumptions")
        
            shock_type = st.selectbox(
                "Shock type",
                ["Level shock", "Growth shock"],
                key="shock_type"
            )
        
            shock_start = st.selectbox(
                "Shock starting year",
                baseline.index.tolist(),
                key="shock_start"
            )
        
            shock_size = st.slider(
                "Shock size",
                -10.0,
                10.0,
                0.0,
                step=0.5,
                key="shock_size"
            )
        
            alt = baseline.copy()
        
            if shock_type == "Level shock":
                alt.loc[alt.index >= shock_start] = alt.loc[alt.index >= shock_start] + shock_size
        
            elif shock_type == "Growth shock":
                for i in range(1, len(alt)):
                    if alt.index[i] >= shock_start:
                        alt.iloc[i] = alt.iloc[i-1] * (1 + shock_size / 100)
        
            # ---- Plot ----
            st.markdown("##### 📈 Baseline vs Alternative")
        
            fig, ax = plt.subplots(figsize=(5, 2))
        
            ax.plot(baseline.index, baseline.values, label="Baseline")
            ax.plot(alt.index, alt.values, linestyle="--", label="Alternative")
        
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=5)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

