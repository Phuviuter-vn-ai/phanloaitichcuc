# ==============================================================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import streamlit as st
import pandas as pd
import math
from datetime import date
from streamlit_option_menu import option_menu
import psycopg2
import bcrypt
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys

# ==============================================================================
# CẤU HÌNH TOÀN BỘ ỨNG DỤNG
# ==============================================================================

# --- CẤU HÌNH KẾT NỐI DATABASE ---
DB_CONFIG = {
    "host": "localhost",
    "database": "Crawwebsite",
    "user": "postgres",
    "password": "Phatdzvai1", # <-- THAY MẬT KHẨU CỦA BẠN TẠI ĐÂY NẾU CẦN
    "port": "5432"
}

# --- CẤU HÌNH MÔ HÌNH AI ---
PHOBERT_MODEL_NAME = "vinai/phobert-base"
MODEL_PATH = "model.pth"
NUM_CLASSES = 3
FEATURE_BATCH_SIZE = 8

# ==============================================================================
# PHẦN 1: CÁC LỚP VÀ HÀM LIÊN QUAN ĐẾN MÔ HÌNH AI
# ==============================================================================

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.3):
        super(ConvNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        logits = self.network(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

def extract_text_features(texts, phobert_model, tokenizer, device, batch_size=FEATURE_BATCH_SIZE):
    phobert_model.eval()
    all_features = []
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
        with torch.no_grad():
            model_output = phobert_model(**encoded_input)
            batch_features = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_features.append(batch_features)
    return np.concatenate(all_features, axis=0)

@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
        phobert_model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME).to(device)
        input_size = phobert_model.config.hidden_size
        classification_model = ConvNet(input_size=input_size, output_size=NUM_CLASSES).to(device)
        classification_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        classification_model.eval()
        st.success(f"Đã tải mô hình thành công trên thiết bị: {device}")
        return tokenizer, phobert_model, classification_model, device
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file mô hình tại '{MODEL_PATH}'. Vui lòng đảm bảo file tồn tại.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải mô hình: {e}")
        return None, None, None, None

def predict(text: str, tokenizer, phobert_model, classification_model, device):
    reverse_label_mapping = {0: "Normal", 1: "Positive News", 2: "Negative News"}
    text_features = extract_text_features([text], phobert_model, tokenizer, device)
    with torch.no_grad():
        input_tensor = torch.tensor(text_features, dtype=torch.float32).to(device)
        _, probabilities = classification_model(input_tensor)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = reverse_label_mapping.get(predicted_idx, "Unknown")
        all_probs_list = probabilities.cpu().numpy().flatten().tolist()
        prob_dict = {reverse_label_mapping[i]: prob for i, prob in enumerate(all_probs_list)}
    return predicted_label, prob_dict

# ==============================================================================
# PHẦN 2: CÁC HÀM XỬ LÝ DATABASE (BAO GỒM CẢ HÀM KHỞI TẠO)
# ==============================================================================

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        st.error(f"Lỗi kết nối đến PostgreSQL: {e}")
        print(f"Lỗi kết nối đến PostgreSQL: {e}", file=sys.stderr)
        return None

@st.cache_resource
def initialize_database():
    """
    Tự động tạo bảng và tài khoản admin ban đầu nếu cần.
    """
    conn = get_db_connection()
    if not conn:
        st.error("Không thể khởi tạo CSDL do lỗi kết nối. Ứng dụng không thể tiếp tục.")
        st.stop()
    try:
        with conn.cursor() as cur:
            # --- Tạo bảng 'users' nếu chưa tồn tại ---
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(100) NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'user'
                );
            """)
            
            # --- Tạo bảng 'classified_articles' nếu chưa tồn tại ---
            cur.execute("""
                 CREATE TABLE IF NOT EXISTS classified_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    summary TEXT NOT NULL,
                    news_source VARCHAR(100),
                    label VARCHAR(50) NOT NULL,
                    prediction_date DATE NOT NULL
                );
            """)

            # --- Kiểm tra xem bảng 'users' có trống không ---
            cur.execute("SELECT COUNT(*) FROM users;")
            if cur.fetchone()[0] == 0:
                st.warning("Bảng 'users' đang trống. Đang tạo tài khoản admin mặc định...")
                # Tự động mã hóa mật khẩu 'admin123' và chèn vào
                admin_pass = 'admin123'
                hashed_password = bcrypt.hashpw(admin_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cur.execute(
                    "INSERT INTO users (username, password, role) VALUES (%s, %s, %s);",
                    ('admin', hashed_password, 'admin')
                )
                st.success("Đã tạo tài khoản admin mặc định (admin/admin123).")
            conn.commit()
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi khởi tạo CSDL: {e}")
    finally:
        if conn:
            conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    if not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password, role FROM users WHERE username = %s;", (username,))
            user_data = cur.fetchone()
            if user_data:
                hashed_password_from_db_str, role = user_data
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password_from_db_str.encode('utf-8')):
                    return role
        return None
    finally:
        if conn: conn.close()

def add_new_user(username, password, role):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            sql_query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s);"
            cur.execute(sql_query, (username, hashed_password, role))
            conn.commit()
            st.cache_data.clear()
            return True
    except psycopg2.IntegrityError:
        return False
    finally:
        if conn: conn.close()

@st.cache_data(ttl=600)
def fetch_all_users():
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    try:
        query = "SELECT id, username, role FROM users ORDER BY id ASC;"
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()

def update_user_role(user_id, new_role):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET role = %s WHERE id = %s;", (new_role, user_id))
            conn.commit()
            if cur.rowcount > 0:
                st.cache_data.clear()
                return True
        return False
    finally:
        if conn: conn.close()

def delete_user(user_id):
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
            conn.commit()
            if cur.rowcount > 0:
                st.cache_data.clear()
                return True
        return False
    finally:
        if conn: conn.close()

@st.cache_data(ttl=300)
def fetch_classified_data():
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    try:
        query = "SELECT id, title, summary, news_source, label, prediction_date FROM classified_articles ORDER BY id DESC;"
        df = pd.read_sql_query(query, conn)
        return df
    except (Exception, psycopg2.Error):
        return pd.DataFrame()
    finally:
        if conn: conn.close()

def insert_classified_article(title, summary, label, news_source, prediction_date):
    conn = get_db_connection()
    if not conn: return False
    sql = "INSERT INTO classified_articles (title, summary, label, news_source, prediction_date) VALUES (%s, %s, %s, %s, %s);"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (title, summary, label, news_source, prediction_date))
            conn.commit()
            st.cache_data.clear()
            return True
    finally:
        if conn: conn.close()

def update_article_label(article_id, new_label):
    conn = get_db_connection()
    if not conn: return False
    sql = "UPDATE classified_articles SET label = %s WHERE id = %s;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (new_label, article_id))
            conn.commit()
            if cur.rowcount > 0:
                st.cache_data.clear()
                return True
        return False
    finally:
        if conn: conn.close()

# ==============================================================================
# PHẦN 3: GIAO DIỆN NGƯỜI DÙNG (UI) VÀ LUỒNG CHÍNH CỦA ỨNG DỤNG
# ==============================================================================

def render_login_page():
    st.title("Đăng nhập Hệ thống")
    
    # --- Form Đăng nhập ---
    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập", use_container_width=True, type="primary"):
        role = verify_user(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_role = role
            st.rerun()
        else:
            st.error("Tên đăng nhập hoặc mật khẩu không chính xác.")

    # --- Form Đăng ký (TÍNH NĂNG MỚI) ---
    with st.expander("Chưa có tài khoản? Đăng ký tại đây"):
        with st.form("registration_form", clear_on_submit=True):
            st.subheader("Tạo tài khoản mới")
            reg_username = st.text_input("Tên đăng nhập mong muốn")
            reg_password = st.text_input("Mật khẩu", type="password")
            reg_confirm_password = st.text_input("Xác nhận lại mật khẩu", type="password")
            
            if st.form_submit_button("Đăng ký"):
                if not reg_username or not reg_password:
                    st.warning("Vui lòng điền đầy đủ tên đăng nhập và mật khẩu.")
                elif reg_password != reg_confirm_password:
                    st.error("Mật khẩu xác nhận không khớp.")
                else:
                    # Vai trò mặc định cho người dùng tự đăng ký là 'user'
                    if add_new_user(reg_username, reg_password, 'user'):
                        st.success(f"Tạo tài khoản '{reg_username}' thành công! Vui lòng đăng nhập.")
                    else:
                        st.error(f"Tên đăng nhập '{reg_username}' đã tồn tại.")

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def display_interactive_list(df: pd.DataFrame):
    # (Code không thay đổi)
    header_cols = st.columns([1, 6, 2, 3])
    header_cols[0].markdown("**ID**")
    header_cols[1].markdown("**Nội dung**")
    header_cols[2].markdown("**Nguồn**")
    header_cols[3].markdown("**Nhãn Phân loại**")
    st.divider()
    if df.empty:
        st.warning("Không có tin tức nào để hiển thị.")
        return
    all_labels = ["Normal", "Positive News", "Negative News"]
    for _, row in df.iterrows():
        row_cols = st.columns([1, 6, 2, 3])
        row_cols[0].write(row['id'])
        with row_cols[1]:
            st.markdown(f"**{row.get('title', 'Không có tiêu đề')}**")
            st.caption(row.get('summary', 'Không có nội dung'))
        row_cols[2].write(row.get('news_source', 'N/A'))
        with row_cols[3]:
            current_label = row.get('label', 'N/A')
            if st.session_state.user_role == 'admin':
                try:
                    current_index = all_labels.index(current_label)
                except ValueError: current_index = 0
                new_label = st.selectbox("Sửa nhãn", options=all_labels, index=current_index, key=f"label_select_{row['id']}", label_visibility="collapsed")
                if new_label != current_label:
                    if update_article_label(row['id'], new_label):
                        st.toast(f"Đã cập nhật nhãn cho ID {row['id']}.", icon="✅")
                        st.rerun()
            else:
                st.info(current_label)
        st.divider()

def render_advanced_pagination(total_pages: int, current_page: int, page_number_key: str):
    # (Code không thay đổi)
    if total_pages <= 1: return
    st.write(f"Trang {current_page} / {total_pages}")
    nav_cols = st.columns(9)
    if nav_cols[0].button("Đầu", disabled=(current_page == 1), key=f"first_{page_number_key}"): st.session_state[page_number_key] = 1; st.rerun()
    if nav_cols[1].button("Trước", disabled=(current_page == 1), key=f"prev_{page_number_key}"): st.session_state[page_number_key] -= 1; st.rerun()
    if nav_cols[7].button("Sau", disabled=(current_page == total_pages), key=f"next_{page_number_key}"): st.session_state[page_number_key] += 1; st.rerun()
    if nav_cols[8].button("Cuối", disabled=(current_page == total_pages), key=f"last_{page_number_key}"): st.session_state[page_number_key] = total_pages; st.rerun()

def render_dashboard_page():
    # (Code không thay đổi)
    st.title("📊 Dashboard - Thống kê Tin tức")
    full_df = fetch_classified_data()
    if full_df is None or full_df.empty:
        st.warning("Chưa có dữ liệu nào được phân loại.")
        return
    label_counts = full_df['label'].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số tin", len(full_df))
    col2.metric("✅ Tin Tích cực", label_counts.get("Positive News", 0))
    col3.metric("❌ Tin Tiêu cực", label_counts.get("Negative News", 0))
    col4.metric("📄 Tin Thường", label_counts.get("Normal", 0))
    st.divider()
    st.subheader("Danh sách tin tức đã phân loại")
    items_per_page = 10
    total_items = len(full_df)
    total_pages = math.ceil(total_items / items_per_page) if total_items > 0 else 1
    current_page = st.session_state.get('dashboard_page_number', 1)
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    paginated_df = full_df.iloc[start_idx:end_idx]
    display_interactive_list(paginated_df)
    render_advanced_pagination(total_pages, current_page, 'dashboard_page_number')

def render_classification_page():
    # (Code không thay đổi)
    st.title("✍️ Phân loại Tin tức Tích cực / Tiêu cực")
    st.markdown("Nhập nội dung vào ô bên dưới để hệ thống phân loại.")
    model_components = load_model()
    if not all(model_components):
        st.error("Không thể tải mô hình phân loại. Chức năng này không hoạt động.")
        return
    text_input = st.text_area("Nội dung cần phân loại:", height=200, placeholder="Ví dụ: 'Giá xăng tăng mạnh gây ảnh hưởng lớn đến đời sống người dân.'")
    if st.button("Phân loại ngay", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung để phân loại.")
        else:
            with st.spinner("Đang phân tích..."):
                label, probabilities = predict(text_input, *model_components)
                st.subheader("Kết quả Phân loại")
                if label == "Positive News": st.success(f"**Kết luận: {label}** (Tin tức Tích cực)")
                elif label == "Negative News": st.error(f"**Kết luận: {label}** (Tin tức Tiêu cực)")
                else: st.info(f"**Kết luận: {label}** (Tin tức Bình thường)")
                insert_classified_article("Phân loại thủ công", text_input, label, "Manual Input", date.today())
                st.toast("Đã lưu kết quả phân loại vào cơ sở dữ liệu.", icon="💾")
                st.subheader("Phân tích xác suất")
                prob_df = pd.DataFrame(list(probabilities.items()), columns=['Nhãn', 'Xác suất'])
                prob_df['Xác suất'] = prob_df['Xác suất'] * 100
                st.dataframe(prob_df.style.format({'Xác suất': '{:.2f}%'}).background_gradient('Greens', subset=['Xác suất']), use_container_width=True)

def render_user_management_page():
    # (Code không thay đổi)
    st.title("👥 Quản lý Tài khoản")
    if st.session_state.user_role != 'admin':
        st.error("Chỉ có quản trị viên (admin) mới có quyền truy cập trang này.")
        return
    with st.expander("Thêm người dùng mới", expanded=False):
        with st.form("new_user_form", clear_on_submit=True):
            st.subheader("Tạo tài khoản")
            new_username = st.text_input("Tên đăng nhập mới")
            new_password = st.text_input("Mật khẩu mới", type="password")
            new_role = st.selectbox("Vai trò", options=['user', 'admin'], index=0)
            if st.form_submit_button("Thêm người dùng"):
                if new_username and new_password:
                    if add_new_user(new_username, new_password, new_role):
                        st.success(f"Đã thêm người dùng '{new_username}'.")
                    else: st.error(f"Tên người dùng '{new_username}' đã tồn tại.")
                else: st.warning("Tên đăng nhập và mật khẩu không được để trống.")
    st.divider()
    st.subheader("Danh sách người dùng")
    users_df = fetch_all_users()
    if users_df.empty:
        st.info("Không có người dùng nào trong hệ thống.")
    else:
        header_cols = st.columns([1, 3, 2, 2])
        header_cols[0].markdown("**ID**")
        header_cols[1].markdown("**Tên đăng nhập**")
        header_cols[2].markdown("**Vai trò**")
        header_cols[3].markdown("**Hành động**")
        st.markdown("---")
        for _, row in users_df.iterrows():
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            col1.write(row['id'])
            col2.write(row['username'])
            is_current_user = (row['username'] == st.session_state.username)
            with col3:
                current_role = row['role']
                new_role = st.selectbox("Vai trò", options=['user', 'admin'], index=0 if current_role == 'user' else 1, key=f"role_select_{row['id']}", label_visibility="collapsed", disabled=is_current_user)
            with col4:
                btn_cols = st.columns(2)
                if btn_cols[0].button("Cập nhật", key=f"update_{row['id']}", disabled=is_current_user):
                    if update_user_role(row['id'], new_role):
                        st.toast(f"Cập nhật vai trò cho '{row['username']}' thành công.", icon="✅")
                        st.rerun()
                if btn_cols[1].button("Xóa", key=f"delete_{row['id']}", disabled=is_current_user):
                    if delete_user(row['id']):
                        st.toast(f"Đã xóa người dùng '{row['username']}'.", icon="🗑️")
                        st.rerun()
            st.markdown("---")

# ==============================================================================
# LUỒNG CHÍNH CỦA ỨNG DỤNG
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Hệ thống Phân loại Tin tức")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'dashboard_page_number' not in st.session_state:
        st.session_state.dashboard_page_number = 1
    
    # Chạy hàm khởi tạo CSDL một lần duy nhất khi ứng dụng bắt đầu
    initialize_database()

    if not st.session_state.logged_in:
        render_login_page()
    else:
        with st.sidebar:
            st.header(f"Xin chào, {st.session_state.username}!")
            st.caption(f"Vai trò: {st.session_state.user_role}")
            st.divider()
            menu_options = ["Dashboard", "Phân loại Tin tức"]
            menu_icons = ["house-door-fill", "badge-ad-fill"]
            if st.session_state.user_role == 'admin':
                menu_options.append("Quản lý Tài khoản")
                menu_icons.append("people-fill")
            selected_page = option_menu(menu_title="Chức năng", options=menu_options, icons=menu_icons, menu_icon="list", default_index=0)
            st.divider()
            if st.button("Đăng xuất", use_container_width=True):
                logout()

        if selected_page == "Dashboard":
            render_dashboard_page()
        elif selected_page == "Phân loại Tin tức":
            render_classification_page()
        elif selected_page == "Quản lý Tài khoản":
            render_user_management_page()

if __name__ == "__main__":
    main()