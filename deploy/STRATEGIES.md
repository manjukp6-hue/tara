# TARA Universal Release Strategies Guide (4 Strategies)

How to deploy updates to your running TARA systems without downtime:

---

## 1. Rolling Deployment (Zero Downtime)
* **Concept**: Update servers/containers one by one.
* **How it works with TARA**:
  - While Server A is updating, TARA client probing automatically detects Server B is healthy and routes all queries to Server B.
  - Once Server A is updated and reports healthy `/health`, update Server B.
  - Result: 0 seconds downtime for users.

---

## 2. Blue-Green Deployment (Instant Safe Switch)
* **Concept**: Maintain two identical production environments: "Blue" (Active) and "Green" (New).
* **How it works with TARA**:
  - Blue: `https://tara-v1.example.com` (currently in `endpoints.txt`).
  - Deploy new version to Green: `https://tara-v2.example.com`.
  - Test Green privately.
  - Once verified, simply update `endpoints.txt`:
    ```text
    https://tara-v2.example.com
    ```
  - All TARA clients instantly switch to the Green environment on their next automatic 60s refresh.

---

## 3. Canary Deployment (Gradual Verification)
* **Concept**: Test the new release with a portion of clients before 100% rollout.
* **How it works with TARA**:
  - Keep the existing stable provider in `endpoints.txt`.
  - Add the new canary provider to `endpoints.txt`:
    ```text
    https://stable-tara.example.com
    https://canary-tara.example.com
    ```
  - TARA's concurrent probing engine evaluates both live.
  - If Canary fails or has high latency, TARA automatically fails over to the stable provider without user impact.

---

## 4. Recreate Deployment (Clean Restart)
* **Concept**: Terminate the old process and launch the new version.
* **How it works with TARA**:
  - Run `systemctl restart tara` (on Linux VPS) or stop and rerun `app.py`.
  - Best for local development, single-node personal systems, or maintenance windows.
