#!/usr/bin/env bash
# Capture Part D / E kubectl evidence. Run after the cluster is up and the
# manifests are applied. Writes transcripts to screenshots/logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGS="$PWD/screenshots/logs"
mkdir -p "$LOGS"

scrub () { sed -i.bak "s|$HOME|~|g" "$LOGS"/D*.txt "$LOGS"/E*.txt 2>/dev/null || true; rm -f "$LOGS"/*.bak; }

banner () { printf '\n\033[1;36m%s\033[0m\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 ${#1}))"; }

{
  echo "# kubectl cluster-info"
  kubectl cluster-info
  echo
  echo "# kubectl get nodes -o wide"
  kubectl get nodes -o wide
} | tee "$LOGS/D0_cluster.txt"

banner "D1  ConfigMap + Secret"
{
  echo "# kubectl get configmap,secret"
  kubectl get configmap btc-forecast-config -o yaml
  echo "---"
  kubectl get secret btc-forecast-secret -o yaml
} | tee "$LOGS/D1_configmap_secret.txt"

banner "D2  PVC"
kubectl get pvc btc-forecast-pvc -o yaml | tee "$LOGS/D2_pvc.txt"

banner "D3  Deployment + Pods"
{
  echo "# kubectl get deploy,pods -o wide"
  kubectl get deploy,pods -l app=btc-forecast-api -o wide
  echo
  echo "# kubectl describe deploy btc-forecast-api (probes + resources)"
  kubectl describe deploy btc-forecast-api
} | tee "$LOGS/D3_deployment.txt"

banner "D4  Services"
kubectl get svc -l app=btc-forecast-api -o wide | tee "$LOGS/D4_service.txt"
echo "---" | tee -a "$LOGS/D4_service.txt"
kubectl describe svc btc-forecast-api btc-forecast-api-nodeport | tee -a "$LOGS/D4_service.txt"

banner "D5  HPA"
kubectl get hpa btc-forecast-hpa -o yaml | tee "$LOGS/D5_hpa.txt"

banner "D6  Ingress"
{
  kubectl get ingress btc-forecast-ingress -o yaml
  echo
  kubectl describe ingress btc-forecast-ingress
} | tee "$LOGS/D6_ingress.txt"

banner "summary"
{
  echo "# kubectl get pods,svc,hpa,ingress,pvc"
  kubectl get pods,svc,hpa,ingress,pvc -l app.kubernetes.io/part-of=btc-forecast-mlops
  echo
  kubectl get pods,svc,hpa,ingress,pvc
} | tee "$LOGS/D_summary.txt"

scrub
echo "wrote transcripts under $LOGS"
