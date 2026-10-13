#!/bin/bash
set -e

# Configuration
PROJECT_NAME="pulpfictionchatbot"
# Les ID de projet doivent être uniques mondialement. On ajoute un suffixe aléatoire.
PROJECT_ID="pulpfictionchatbot-${RANDOM}"
BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:?Définir BILLING_ACCOUNT_ID (gcloud billing accounts list)}"
REGION="europe-west4" # Correspond à la région définie dans train-vertex.yml

# WIF and SA config
GITHUB_ORG="${GITHUB_ORG:?Définir GITHUB_ORG (propriétaire du repo GitHub)}"
GITHUB_REPO="${GITHUB_REPO:?Définir GITHUB_REPO (nom du repo GitHub)}"
SA_NAME="github-actions-sa"
WIF_POOL="github-actions-pool"
WIF_PROVIDER="github-actions-provider"
BUCKET_NAME="${PROJECT_ID}-staging"

echo "========================================"
echo "Création du projet GCP: $PROJECT_ID"
echo "========================================"

gcloud projects create $PROJECT_ID --name="$PROJECT_NAME"
gcloud config set project $PROJECT_ID

echo "Liaison du compte de facturation..."
gcloud beta billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT_ID

echo "Activation des APIs (Vertex AI, IAM, Storage)..."
# Peut prendre 1 ou 2 minutes
gcloud services enable aiplatform.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  storage.googleapis.com \
  --project $PROJECT_ID

echo "Création du bucket de staging ($BUCKET_NAME)..."
gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION --project=$PROJECT_ID

echo "Création du Service Account ($SA_NAME)..."
gcloud iam service-accounts create $SA_NAME \
  --display-name="GitHub Actions Service Account for Vertex AI" \
  --project=$PROJECT_ID

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Attribution des rôles au Service Account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user" > /dev/null

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin" > /dev/null

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.admin" > /dev/null

echo "Création du Pool Workload Identity ($WIF_POOL)..."
gcloud iam workload-identity-pools create $WIF_POOL \
  --location="global" \
  --description="Pool for GitHub Actions" \
  --display-name="GitHub Actions Pool" \
  --project=$PROJECT_ID

echo "Création du Provider Workload Identity ($WIF_PROVIDER)..."
gcloud iam workload-identity-pools providers create-oidc $WIF_PROVIDER \
  --location="global" \
  --workload-identity-pool=$WIF_POOL \
  --display-name="GitHub Actions Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --project=$PROJECT_ID

PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
WIF_PROVIDER_FULL="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"

echo "Liaison entre GitHub Actions et le Service Account via WIF..."
# Cette ligne est cruciale: elle limite l'usurpation d'identité à ton repo exact
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}" \
  --project=$PROJECT_ID > /dev/null

echo ""
echo "=========================================================="
echo "✅ Configuration terminée ! Voici tes 4 GitHub Secrets :"
echo "=========================================================="
echo "GCP_PROJECT:          $PROJECT_ID"
echo "GCP_SA_EMAIL:         $SA_EMAIL"
echo "GCP_STAGING_BUCKET:   gs://$BUCKET_NAME"
echo "GCP_WIF_PROVIDER:     $WIF_PROVIDER_FULL"
echo "=========================================================="
