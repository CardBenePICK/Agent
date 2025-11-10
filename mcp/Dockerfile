FROM python:3.11-slim

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /mcp

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /mcp

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001","--reload"] 
# uvicorn main:app --port 8001 --reload # 00 폴더에 있는 (mcp server 열기 위해)
# elk 서버도 열어줘야 함.
# docker run --rm -it -p 8000:8000 -v .:/app --name myapp myapp:latest