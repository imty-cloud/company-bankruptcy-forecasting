# Demo dự báo phá sản doanh nghiệp (Streamlit)

Chạy trên máy:
    pip install -r requirements.txt
    streamlit run app.py

Deploy miễn phí lên Streamlit Community Cloud:
1. Tải toàn bộ thư mục này vào repo GitHub (giữ nguyên cấu trúc thư mục).
2. Vào share.streamlit.io, đăng nhập bằng GitHub, chọn "Create app".
3. Chọn repo, branch, Main file path = app.py (hoặc demo_app/app.py nếu để trong thư mục con), bấm Deploy.

Tái tạo mô hình: build_models.py huấn luyện lại 40 mô hình từ siêu tham số Bảng 2 (cần 1year.arff ... 5year.arff);
export_shap.py tính SHAP. Mô hình lưu trong models/, dữ liệu test và dự báo trong data/.
