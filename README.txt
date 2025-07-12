SemanticCacheRAG
1. Giới thiệu 
Dự án này là một phiên bản triển khai thực tế của hệ thống Semantic Caching được đề xuất trong bài báo khoa học: "Semantic Caching of Contextual Summaries for Efficient Question-Answering with Language Models".

Mục tiêu chính là xây dựng một hệ thống Hỏi-Đáp (Question-Answering) thông minh, có khả năng giảm thiểu độ trễ và chi phí tính toán bằng cách tránh các lệnh gọi dư thừa đến Mô hình Ngôn ngữ Lớn (LLM).

Hệ thống hoạt động dựa trên nguyên tắc "cache-first": kiểm tra một bộ nhớ cache ngữ nghĩa được xây dựng trên Redis trước khi thực hiện quy trình RAG (Retrieval-Augmented Generation) đầy đủ.

Link bài báo gốc: https://arxiv.org/pdf/2505.11271

2. Công nghệ được sử dụng
Hệ thống được xây dựng từ các thành phần hiện đại và mạnh mẽ:

Giao diện Người dùng: Streamlit

Bộ não AI (LLM): Google Gemini

Bộ nhớ Cache Ngữ nghĩa: Redis (với module RediSearch)

Bộ khung Kết nối: LangChain

Mô hình Embedding: all-MiniLM-L6-v2

Triển khai: Docker

3. Các tính năng chính 
Cache Ngữ nghĩa: Lưu trữ các cặp câu hỏi, bản tóm tắt theo ngữ cảnh, và câu trả lời cuối cùng vào Redis Hash để tái sử dụng.

Tìm kiếm Vector: Sử dụng RediSearch để tạo chỉ mục vector, cho phép tìm kiếm các câu hỏi tương tự dựa trên ý nghĩa (semantic similarity) thay vì từ khóa.

Hỗ trợ Embedding: Sử dụng mô hình all-MiniLM-L6-v2 (384 chiều) để tạo vector ngữ nghĩa cho các câu hỏi.

Hai chế độ hoạt động:

Chế độ RAG: Hỏi-đáp chuyên sâu dựa trên nội dung của một tài liệu PDF được tải lên.

Chế độ Chatbot chung: Trò chuyện thông thường dựa trên kiến thức của Gemini, đồng thời vẫn tận dụng và làm giàu cho cache.

Triển khai với Docker: Toàn bộ hệ thống, bao gồm cả Redis, có thể được khởi chạy dễ dàng bằng một lệnh docker-compose.

4. Hướng dẫn
Yêu cầu hệ thống
Docker và Docker Compose

Python 3.10+

Một API Key của Google Gemini

Các bước cài đặt
Clone repository này:

git clone [URL_CUA_REPO_CUA_BAN]
cd [TEN_REPO_CUA_BAN]

Tạo môi trường ảo và cài đặt các thư viện:

python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

Cấu hình API Key:

Tạo một file có tên là .env.

Thêm vào đó API key của bạn như sau:

MY_GEMINI_API_KEY="..................."

Khởi chạy hệ thống với Docker:

Đảm bảo Docker Desktop đang chạy.

Chạy lệnh sau để khởi động Redis và Streamlit:

docker-compose up --build



5. Cách sử dụng
Chế độ Chatbot: Ngay khi khởi động, bạn có thể bắt đầu trò chuyện. Các câu hỏi và câu trả lời sẽ tự động được lưu vào cache.

Chế độ RAG:

Sử dụng thanh bên trái (sidebar) để tải lên một file PDF.

Nhấn nút "Xử lý tài liệu".

Sau khi xử lý xong, bạn có thể bắt đầu đặt các câu hỏi liên quan đến nội dung của file PDF.