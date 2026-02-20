targetScope = 'resourceGroup'

@description('Deployment location')
param location string = resourceGroup().location

@description('Static Web App location (must be one of the regions supported by Microsoft.Web/staticSites)')
param staticWebAppLocation string = 'eastus2'

@description('PostgreSQL location (must be allowed for Microsoft.DBforPostgreSQL/flexibleServers in your subscription)')
param postgresLocation string = 'eastus2'

@description('Environment name (for resource naming)')
@minLength(1)
param environmentName string

@description('Backend container image (set by CI/CD or azd)')
param backendImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Container port exposed by FastAPI app')
param backendTargetPort int = 8001

@description('PostgreSQL admin username')
param postgresAdminUsername string = 'pgadminuser'

@secure()
@description('PostgreSQL admin password')
param postgresAdminPassword string

@secure()
@description('JWT secret for backend token validation')
param authJwtSecret string

@description('PostgreSQL database name')
param postgresDatabaseName string = 'searchadlearning'

var nameSuffix = toLower(replace('${environmentName}${uniqueString(resourceGroup().id)}', '-', ''))
var postgresNameSuffix = toLower(replace('${environmentName}${uniqueString(resourceGroup().id, postgresLocation)}', '-', ''))
var appInsightsName = take('appi-${nameSuffix}', 24)
var logAnalyticsName = take('law-${nameSuffix}', 63)
var keyVaultName = take('kv-${nameSuffix}', 24)
var storageAccountName = take(replace('st${nameSuffix}', '-', ''), 24)
var containerRegistryName = take(replace('cr${nameSuffix}', '-', ''), 50)
var postgresServerName = take('pg-${postgresNameSuffix}', 63)
var containerAppsEnvName = take('cae-${nameSuffix}', 32)
var backendAppName = take('ca-backend-${nameSuffix}', 32)
var frontendAppName = take('swa-${nameSuffix}', 60)

module monitoring './modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    location: location
    appInsightsName: appInsightsName
    logAnalyticsName: logAnalyticsName
  }
}

module storage './modules/storage.bicep' = {
  name: 'storage'
  params: {
    location: location
    storageAccountName: storageAccountName
  }
}

module acr './modules/acr.bicep' = {
  name: 'acr'
  params: {
    location: location
    registryName: containerRegistryName
  }
}

module keyvault './modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    location: location
    keyVaultName: keyVaultName
  }
}

module postgres './modules/postgres.bicep' = {
  name: 'postgres'
  params: {
    location: postgresLocation
    serverName: postgresServerName
    adminUsername: postgresAdminUsername
    adminPassword: postgresAdminPassword
    databaseName: postgresDatabaseName
  }
}

module backend './modules/containerapp.bicep' = {
  name: 'backend'
  params: {
    location: location
    containerAppsEnvironmentName: containerAppsEnvName
    containerAppName: backendAppName
    environmentName: environmentName
    serviceName: 'backend'
    registryServer: acr.outputs.registryLoginServer
    backendImage: backendImage
    backendTargetPort: backendTargetPort
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    logAnalyticsWorkspaceResourceId: monitoring.outputs.logAnalyticsWorkspaceResourceId
    appInsightsConnectionString: monitoring.outputs.appInsightsConnectionString
    databaseUrlSecretValue: 'postgresql+psycopg://${postgresAdminUsername}:${postgresAdminPassword}@${postgres.outputs.postgresFqdn}:5432/${postgresDatabaseName}?sslmode=require'
    authJwtSecretValue: authJwtSecret
  }
}

module frontend './modules/staticwebapp.bicep' = {
  name: 'frontend'
  params: {
    location: staticWebAppLocation
    staticWebAppName: frontendAppName
    environmentName: environmentName
    serviceName: 'frontend'
  }
}

output AZURE_LOCATION string = location

output SERVICE_BACKEND_NAME string = backend.outputs.backendName
output SERVICE_BACKEND_RESOURCE_ID string = backend.outputs.backendResourceId
output SERVICE_BACKEND_URI string = backend.outputs.backendUrl

output SERVICE_FRONTEND_NAME string = frontend.outputs.frontendName
output SERVICE_FRONTEND_RESOURCE_ID string = frontend.outputs.frontendResourceId
output SERVICE_FRONTEND_URI string = frontend.outputs.frontendUrl

output KEYVAULT_NAME string = keyvault.outputs.keyVaultName
output STORAGE_ACCOUNT_NAME string = storage.outputs.storageAccountName
output POSTGRES_SERVER_FQDN string = postgres.outputs.postgresFqdn
output POSTGRES_DATABASE_NAME string = postgresDatabaseName
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = acr.outputs.registryLoginServer
output AZURE_CONTAINER_REGISTRY_NAME string = acr.outputs.registryName
