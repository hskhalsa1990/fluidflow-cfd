FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
RUN useradd --create-home runner && mkdir -p /app/outputs && chown runner /app/outputs
USER runner
ENTRYPOINT ["python", "main.py"]
CMD ["cavity", "--re", "100", "--n", "65", "--out", "/app/outputs"]
