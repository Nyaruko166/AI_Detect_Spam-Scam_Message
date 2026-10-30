import os
import sys

import joblib


def find_model_dir():
    """Find the directory where the Logistic Regression model artifacts are exported."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "model"),
        os.path.join(script_dir, "..", "..", "model", "Logistic Regression"),
        os.path.join(script_dir, "..", "..", "model"),
        os.path.abspath(os.path.join("src", "Logistic Regression", "model")),
        os.path.abspath(os.path.join("model", "Logistic Regression")),
        os.path.abspath("model"),
    ]
    for path in candidates:
        if os.path.isdir(path) and (
            os.path.exists(os.path.join(path, "pipeline.joblib"))
            or os.path.exists(os.path.join(path, "model.joblib"))
        ):
            return path
    return None


def load_model():
    """Load exported model pipeline or individual model and vectorizer."""
    model_dir = find_model_dir()
    if not model_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(script_dir, "model")
        raise FileNotFoundError(
            f"Could not locate exported model files in '{expected_dir}'. "
            "Please run 'src/Logistic Regression/main.py' first to train and export the model."
        )

    pipeline_path = os.path.join(model_dir, "pipeline.joblib")
    if os.path.exists(pipeline_path):
        print(f"Loading pipeline from: {pipeline_path}")
        return joblib.load(pipeline_path)

    # Fallback to model + vectorizer
    model_path = os.path.join(model_dir, "model.joblib")
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    print(f"Loading model from: {model_path}")
    print(f"Loading vectorizer from: {vectorizer_path}")
    from sklearn.pipeline import Pipeline

    model = joblib.load(model_path)
    vec = joblib.load(vectorizer_path)
    return Pipeline([("vectorizer", vec), ("classifier", model)])


def run_tests():
    pipeline = load_model()

    # Array of sample test messages (Legitimate / Ham and Scam / Spam)
    sample_messages = [
        # --- Phishing / Scam Messages ---
        {
            "category": "Expected SPAM",
            "text": "Chúc mừng bạn đã trúng thưởng xe máy SH 150i và 100 triệu đồng trong chương trình tri ân khách hàng. Nhấp vào link http://trian-sh.vip để nhận quà ngay.",
        },
        {
            "category": "Expected SPAM",
            "text": "Vietcombank canh bao: Tai khoan cua ban co giao dich bat thuong 20,000,000 VND. Vui long dang nhap https://vcb-digibank-secure.com de huy giao dich.",
        },
        {
            "category": "Expected SPAM",
            "text": "Tuyển CTV xử lý đơn hàng Shopee tại nhà. Thu nhập 500k-1tr/ngày, không cần vốn, nhận lương theo ngày. Liên hệ Zalo: [PHONE].",
        },
        {
            "category": "Expected SPAM",
            "text": "Cuc Canh Sat Giao Thong: Bien so xe cua ban bi phat nguoi vi pham toc do. Dong phat tai http://csgt-phatnguoi.gov-vn.info truoc ngay mai.",
        },
        {
            "category": "Expected SPAM",
            "text": "Goi vay tieu dung nhanh khong can chung minh thu nhap, giai ngan trong 15 phut lai suat 0%. Nhan tin theo cu phap VAY gui [PHONE].",
        },
        # --- Legitimate / Ham Messages ---
        {
            "category": "Expected HAM",
            "text": "Chiều nay tan học nhóm mình đi uống trà sữa và làm bài tập lớn môn TTNT không các cậu?",
        },
        {
            "category": "Expected HAM",
            "text": "Mẹ ơi tối nay con tăng ca về muộn khoảng 8h, mẹ và cả nhà cứ ăn cơm trước nhé.",
        },
        {
            "category": "Expected HAM",
            "text": "Mã OTP xác thực giao dịch chuyển tiền của quý khách là 849201. Mã có hiệu lực trong 2 phút. Tuyệt đối không chia sẻ mã này cho bất kỳ ai.",
        },
        {
            "category": "Expected HAM",
            "text": "Chào anh Tuấn, em gửi anh slide báo cáo tiến độ đồ án tuần này qua email rồi nhé, anh xem giúp em.",
        },
        {
            "category": "Expected HAM",
            "text": "Bác ơi tí về nhớ mua giúp cháu bịch bột giặt với chai dầu ăn nhé, cháu cảm ơn bác nhiều.",
        },
    ]

    texts = [sample["text"] for sample in sample_messages]
    predictions = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)

    print("\n" + "=" * 90)
    print(
        f"{'#':<3} | {'EXPECTED':<14} | {'PREDICTION':<18} | {'SPAM PROB':<10} | {'MESSAGE CONTENT'}"
    )
    print("=" * 90)

    for idx, (sample, pred, prob) in enumerate(
        zip(sample_messages, predictions, probabilities), start=1
    ):
        label = "SPAM / SCAM" if pred == 1 else "HAM (Legitimate)"
        spam_prob = prob[1] * 100
        category = sample["category"]

        # Truncate text for table overview
        preview_text = (
            sample["text"] if len(sample["text"]) <= 42 else sample["text"][:39] + "..."
        )
        print(
            f"{idx:<3} | {category:<14} | {label:<18} | {spam_prob:>6.2f}%    | {preview_text}"
        )

    print("=" * 90)

    # Detailed Cards
    print("\n--- DETAILED TEST RESULTS ---")
    for idx, (sample, pred, prob) in enumerate(
        zip(sample_messages, predictions, probabilities), start=1
    ):
        status = "SPAM / SCAM" if pred == 1 else "HAM (Legitimate)"
        spam_prob = prob[1] * 100
        ham_prob = prob[0] * 100

        print(f"\n[Test #{idx}] ({sample['category']})")
        print(f'Message : "{sample["text"]}"')
        print(f"Result  : {status}")
        print(
            f"Results  : Ham Probability = {ham_prob:.2f}% | Spam Probability = {spam_prob:.2f}%"
        )


if __name__ == "__main__":
    run_tests()
