FROM python:3.11-slim

WORKDIR /app

# nginx fronts the app; gettext-base provides envsubst for the config template.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx gettext-base \
    && rm -f /etc/nginx/sites-enabled/default \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY nginx.conf.template /app/nginx.conf.template
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["/app/entrypoint.sh"]
