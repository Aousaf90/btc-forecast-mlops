"""Locust load profile for the BTC forecast API (Part D5).

Prophet inference is CPU-bound, so a small number of concurrent /predict
calls is enough to push the HPA past its 60% CPU target. /health is mixed
in so the run also exercises the probe path under load.

    locust -f scripts/locustfile.py --headless \
           -u 20 -r 5 -t 3m --host http://127.0.0.1:5000
"""

from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(5)
    def predict(self):
        self.client.post(
            "/predict",
            json={"forecast_days": 7},
            headers={"Content-Type": "application/json"},
            name="/predict",
        )

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")
