# Dùng bản Python 3.9 slim (bản này ổn định với các thư viện AI cũ)
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements và cài đặt
COPY requirements.txt .

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Cài đặt thư viện (có timeout dài để tránh lỗi mạng)
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy toàn bộ code vào
COPY . .

# Lệnh chạy server
CMD ["python", "main.py"]