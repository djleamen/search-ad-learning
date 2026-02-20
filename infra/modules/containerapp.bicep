targetScope = 'resourceGroup'

param location string
param containerAppsEnvironmentName string
param containerAppName string
param environmentName string
param serviceName string
param registryServer string
param backendImage string
param backendTargetPort int
param logAnalyticsWorkspaceId string
param logAnalyticsWorkspaceResourceId string
param appInsightsConnectionString string
param databaseUrlSecretUri string
param authJwtSecretUri string

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspaceId
        sharedKey: listKeys(logAnalyticsWorkspaceResourceId, '2023-09-01').primarySharedKey
      }
    }
  }
}

resource backendApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  tags: {
    'azd-env-name': environmentName
    'azd-service-name': serviceName
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      registries: [
        {
          server: registryServer
          identity: 'system'
        }
      ]
      ingress: {
        external: true
        targetPort: backendTargetPort
        transport: 'auto'
      }
      activeRevisionsMode: 'Single'
      secrets: [
        {
          name: 'database-url'
          keyVaultUrl: databaseUrlSecretUri
          identity: 'system'
        }
        {
          name: 'auth-jwt-secret'
          keyVaultUrl: authJwtSecretUri
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: backendImage
          env: [
            {
              name: 'DATABASE_URL'
              secretRef: 'database-url'
            }
            {
              name: 'AUTH_JWT_SECRET'
              secretRef: 'auth-jwt-secret'
            }
            {
              name: 'APPINSIGHTS_CONNECTION_STRING'
              value: appInsightsConnectionString
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

output backendName string = backendApp.name
output backendResourceId string = backendApp.id
output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output backendPrincipalId string = backendApp.identity.principalId
