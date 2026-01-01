import pandas as pd
import psycopg2
import os

# --- CẤU HÌNH KẾT NỐI DATABASE ---
# Sử dụng lại cấu hình từ script trước của bạn.
# Hãy đảm bảo các thông tin này là chính xác.
db_config = {
    "host": "localhost",
    "database": "Crawwebsite",  # Cùng database với script crawl
    "user": "postgres",
    "password": "Phatdzvai1",
    "port": "5432"
}

# --- CẤU HÌNH FILE DỮ LIỆU ---
CSV_FILE_PATH = "Dataset_HOU.csv"
TABLE_NAME = "hou_articles"

def import_csv_to_postgres(csv_path, db_params, table_name):
    """
    Hàm chính để đọc file CSV và nhập dữ liệu vào bảng PostgreSQL.
    """
    # 1. Kiểm tra sự tồn tại của file CSV
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file '{csv_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    print(f"Đang đọc dữ liệu từ '{csv_path}'...")
    try:
        # 2. Đọc và làm sạch dữ liệu bằng Pandas
        df = pd.read_csv(csv_path)
        
        # Chỉ giữ lại các cột cần thiết và loại bỏ các hàng có giá trị rỗng
        df = df[['Title', 'Label']].dropna()
        
        # Làm sạch cột 'Label' tương tự như trong script ML
        df['Label'] = df['Label'].str.replace('"', '').str.strip()
        
        # Loại bỏ các hàng có 'Title' hoặc 'Label' rỗng sau khi làm sạch
        df = df[df['Title'].str.strip() != '']
        df = df[df['Label'].str.strip() != '']

        print(f"Đã đọc và làm sạch thành công {len(df)} dòng dữ liệu.")
        
    except Exception as e:
        print(f"Lỗi khi đọc hoặc xử lý file CSV: {e}")
        return

    conn = None
    try:
        # 3. Kết nối tới PostgreSQL
        print(f"\nĐang kết nối tới database '{db_params['database']}'...")
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        print("Kết nối thành công!")

        # 4. Tạo bảng nếu chưa tồn tại
        print(f"Kiểm tra và tạo bảng '{table_name}' nếu cần...")
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                title TEXT UNIQUE NOT NULL,
                label VARCHAR(50)
            );
        """
        cur.execute(create_table_sql)
        print("Kiểm tra bảng hoàn tất.")

        # 5. Chèn dữ liệu vào bảng
        print(f"Bắt đầu chèn {len(df)} dòng vào bảng. Các dòng trùng lặp (dựa trên tiêu đề) sẽ được bỏ qua.")
        insert_count = 0
        
        # Vòng lặp qua từng dòng trong DataFrame để chèn vào DB
        for index, row in df.iterrows():
            title = row['Title']
            label = row['Label']
            
            try:
                # Sử dụng ON CONFLICT để bỏ qua nếu title đã tồn tại
                insert_query = f"""
                    INSERT INTO {table_name} (title, label)
                    VALUES (%s, %s)
                    ON CONFLICT (title) DO NOTHING;
                """
                cur.execute(insert_query, (title, label))
                
                # cur.rowcount sẽ trả về 1 nếu chèn thành công, 0 nếu bị bỏ qua
                if cur.rowcount > 0:
                    insert_count += 1

            except psycopg2.Error as e:
                print(f"  Lỗi khi chèn dòng {index + 1}: {title[:50]}... Lỗi: {e}")
                conn.rollback() # Hoàn tác lại nếu có lỗi

        # 6. Commit và đóng kết nối
        conn.commit()
        cur.close()
        
        print("\n--- HOÀN TẤT ---")
        print(f"Đã chèn thành công {insert_count} dòng mới.")
        print(f"{len(df) - insert_count} dòng đã tồn tại trong database và được bỏ qua.")

    except psycopg2.Error as e:
        print(f"\nLỗi liên quan đến PostgreSQL: {e}")
        print("Vui lòng kiểm tra lại thông tin kết nối (host, database, user, password).")

    finally:
        if conn is not None:
            conn.close()
            print("Đã đóng kết nối database.")

# --- CHẠY CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    import_csv_to_postgres(CSV_FILE_PATH, db_config, TABLE_NAME)