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
from io import BytesIO
import re
from packaging.version import Version
LooseVersion = Version

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
    st.markdown("### 📊 Thông tin mô tả")
    st.dataframe(df_desc, height=180, use_container_width=True)


# ==========================================================
# ▶️ TẢI & HIỂN THỊ DỮ LIỆU
# ==========================================================
if st.button("🚀 Tải & Hiển thị dữ liệu"):

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

    # ----------------------------
    # 📋 Hiển thị dữ liệu
    # ----------------------------
    #st.dataframe(df_all)
    # Chuẩn hoá hiển thị cột thời gian
    df_display = df_all.copy()
    if isinstance(df_display.index, pd.DatetimeIndex):
        df_display.index = df_display.index.strftime("%Y-%m-%d")

    st.markdown("### 📊 Dữ liệu")
    st.dataframe(df_display, height=180, use_container_width=True)
    
    st.markdown("### 🧾 Thống kê mô tả dữ liệu")
    st.dataframe(df_display.describe(), height=180, use_container_width=True)

    st.sidebar.success(f"✅ Tổng {len(df_all.columns)} cột dữ liệu đã tải.")




# ==========================================================
# 📊 VẼ DỮ LIỆU (có tuỳ chỉnh thời gian & tần suất)
# ==========================================================
if "df_all" in st.session_state:
    df_plot = st.session_state["df_all"]
    st.subheader("📊 Tùy chỉnh và Vẽ biểu đồ")

    # ========== 🕒 Chọn khoảng thời gian ==========
    st.markdown("### 🕒 Lọc thời gian hiển thị")
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
    st.markdown("### ⏳ Chọn tần suất vẽ dữ liệu")
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
    st.markdown("### 🧩 Chọn chỉ tiêu hiển thị")
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
        ax.set_ylabel(", ".join([left_names[c] for c in left_cols]), color="tab:blue")
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
            ax2.set_ylabel(", ".join([right_names[c] for c in right_cols]), color="tab:red")
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


