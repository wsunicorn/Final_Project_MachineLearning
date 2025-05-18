import os
import cv2
import numpy as np
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
from sklearn.exceptions import NotFittedError
from train_models import CustomSVM  # Import CustomSVM từ train_models.py

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Tải các mô hình và scaler
try:
    scaler = joblib.load('scaler_model.pkl')
    pca = joblib.load('pca_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    custom_svm_model = joblib.load('custom_svm_model.pkl')  # Thêm tải Custom SVM
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file mô hình: {e}")
    scaler = pca = svm_model = rf_model = custom_svm_model = None
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    scaler = pca = svm_model = rf_model = custom_svm_model = None

# Hàm kiểm tra định dạng file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Hàm trích xuất đặc trưng FFT
def extract_features(image_path):
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Không đọc được ảnh: {image_path}")
            return None

        img_resized = cv2.resize(img_gray, (300, 300), interpolation=cv2.INTER_AREA)
        fft_image = np.fft.fft2(img_resized)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.log1p(np.abs(fft_shifted)).flatten()
        return magnitude_spectrum
    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng: {e}")
        return None

# Route trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    if not all([scaler, pca, svm_model, rf_model, custom_svm_model]):
        flash('Lỗi: Các mô hình chưa được tải. Vui lòng chạy train_models.py.')
        return render_template('index.html')

    if request.method == 'POST':
        # Kiểm tra file tải lên
        if 'file' not in request.files:
            flash('Không có file được tải lên!')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file!')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            # Trích xuất đặc trưng
            features = extract_features(file_path)
            if features is None:
                flash('Không thể xử lý ảnh. Vui lòng thử ảnh khác.')
                os.remove(file_path)
                return redirect(request.url)

            # Kiểm tra số chiều đặc trưng
            expected_n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
            if expected_n_features and len(features) != expected_n_features:
                flash(f'Lỗi: Số chiều đặc trưng ({len(features)}) không khớp với mô hình ({expected_n_features}).')
                os.remove(file_path)
                return redirect(request.url)

            # Chuẩn hóa và giảm chiều
            try:
                features_scaled = scaler.transform([features])
                features_pca = pca.transform(features_scaled)
            except NotFittedError:
                flash('Lỗi: StandardScaler chưa được huấn luyện. Vui lòng chạy lại train_models.py.')
                os.remove(file_path)
                return redirect(request.url)
            except ValueError as e:
                flash(f'Lỗi xử lý đặc trưng: {str(e)} (Kiểm tra số chiều đặc trưng).')
                os.remove(file_path)
                return redirect(request.url)

            # Lấy mô hình được chọn
            model_name = request.form.get('model', 'svm')
            if model_name == 'svm':
                model = svm_model
                model_label = 'SVM'
            elif model_name == 'rf':
                model = rf_model
                model_label = 'Random Forest'
            else:
                model = custom_svm_model
                model_label = 'Custom SVM'

            # Dự đoán
            try:
                prediction = model.predict(features_pca)[0]
                probabilities = model.predict_proba(features_pca)[0]
                result = 'Thật' if prediction == 1 else 'Giả'
                prob_fake = probabilities[0] * 100
                prob_real = probabilities[1] * 100
            except AttributeError as e:
                flash(f'Lỗi: Mô hình {model_label} không hỗ trợ dự đoán hoặc xác suất: {e}')
                os.remove(file_path)
                return redirect(request.url)

            image_url = f"uploads/{filename}"

            return render_template('index.html', result=result, 
                                 prob_fake=f"{prob_fake:.2f}%", prob_real=f"{prob_real:.2f}%",
                                 model_used=model_label.upper(), image_url=image_url)
            # Xóa file sau khi xử lý
            os.remove(file_path)

            return render_template('index.html', result=result, 
                                prob_fake=f"{prob_fake:.2f}%", prob_real=f"{prob_real:.2f}%",
                                model_used=model_label.upper())

        else:
            flash('Định dạng file không được hỗ trợ! Chỉ chấp nhận .png, .jpg, .jpeg.')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5003)