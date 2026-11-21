from src.mobile_price import clean_data, train_best_model, evaluate_model


def run_pipeline():
    print("========== BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ==========")
    clean_data()

    print("\n========== BƯỚC 2: HUẤN LUYỆN MÔ HÌNH ==========")
    train_best_model()

    print("\n========== BƯỚC 3: ĐÁNH GIÁ VÀ XUẤT BÁO CÁO ==========")
    evaluate_model()

    print("\n✓ ĐÃ HOÀN THÀNH TOÀN BỘ PIPELINE THÀNH CÔNG!")


if __name__ == '__main__':
    run_pipeline()